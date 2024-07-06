from __future__ import annotations

import math
from typing import TypeVar

import numpy as np

from metalpy.mepa.utils import is_serial
from metalpy.utils.bounds import Bounds
from metalpy.utils.dhash import dhash
from metalpy.utils.polygon import ear_clip, re_intersect, Edge
from metalpy.utils.geometry import gen_random_convex_polygon
from metalpy.utils.numeric import limit_significand
from metalpy.utils.rand import check_random_state
from metalpy.utils.ti_lazy import ti_lazy_kernel, ti
from metalpy.utils.type import Self
from . import Shape3D
from .shape3d import TransformedArray

PrismT = TypeVar('PrismT', bound='Prism')


class Prism(Shape3D):
    EdgesPerProgress = 4  # 每四条边作为一个进度单位

    def __init__(self,
                 pts,
                 z0: float | int = 0,
                 z1: float | int = 1,
                 cells=None,
                 simplify=True,
                 check_non_simple=True,
                 verbose=True
                 ):
        """定义任意底面棱锥

        Parameters
        ----------
        pts
            底面多边形顶点列表
        z0, z1
            底面和顶面高度
        cells
            array(n-2, 3)，每行包含三个属于(0, n-1)的下标，指定pts三角化后的结果
        simplify
            指示是否将多边形简单化。
            默认为 `True` ，此时会自动通过重相交算法尝试检查自交点并转换为简单多边形
        check_non_simple
            指示通过耳切法进行三角化时是否检查非简单多边形。
            为 `True` 时遇到非简单多边形会抛出异常，
            否则会直接返回部分三角化的结果（结果大概率错误）
        verbose
            是否输出辅助信息（主要为顶点数较多时导出PyVista模型会触发进度条）
        """
        super().__init__()
        self.pts = np.asarray(pts)
        self.z0 = min(z0, z1)
        self.z1 = max(z0, z1)

        self.check_non_simple = check_non_simple
        self.simplify = simplify
        self.verbose = verbose

        self._cells = None
        self._pts = None
        self._use_external_cells = False

        if cells is not None:
            self.triangulated_cells = cells

    @classmethod
    def rand(cls: type[PrismT],
             n_edges: int | None = None,
             size=(1, 1),
             z0=0,
             z1=1,
             random_state=None
             ) -> PrismT:
        """生成随机棱柱

        Parameters
        ----------
        n_edges
            随机棱柱的边数
        size
            随机棱柱在xy平面上的尺寸
        z0, z1
            随机棱柱的底面和顶面高度
        random_state
            随机种子

        Returns
        -------
        ret
            随机棱柱实例
        """
        random_state = check_random_state(random_state)

        n_edges = int(random_state.rand() * 18) + 3 if n_edges is None else n_edges  # 随机最多20边
        pts = gen_random_convex_polygon(n_edges, size=size, random_state=random_state)

        return cls(pts, z0, z1)

    @classmethod
    def rand_spiky(cls: type[PrismT],
                   n_edges: int | None = None,
                   r0=0.1,
                   r1=1,
                   z0=0,
                   z1=1,
                   random_state=None
                   ) -> PrismT:
        """生成随机刺挠棱柱

        Parameters
        ----------
        n_edges
            随机棱柱的边数
        r0, r1
            随机的半径范围
        z0, z1
            随机棱柱的底面和顶面高度
        random_state
            随机种子

        Returns
        -------
        ret
            随机刺挠棱柱实例
        """
        random_state = check_random_state(random_state)

        n_edges = int(random_state.rand() * 18) + 3 if n_edges is None else n_edges  # 随机最多20边

        ratios = np.sort(random_state.rand(n_edges))
        ratios /= ratios[-1] - ratios[0]
        max_ratio_swept = np.diff(ratios).max()

        # 下界：保证多边形旋转角度不小于180°
        # 上界：保证多边形任意两条边之间旋转角度不大于180°
        theta_min = np.pi
        theta_max = min(2 * np.pi, np.pi / max_ratio_swept)
        theta = random_state.rand() * (theta_max - theta_min) + theta_min
        theta0 = random_state.rand() * (2 * np.pi - theta)
        angles = ratios * theta + theta0

        radii = random_state.rand(n_edges) * (r1 - r0) + r0

        pts = np.c_[np.cos(angles), np.sin(angles)] * radii[:, None]

        return cls(pts, z0, z1)

    def do_place(self, mesh_cell_centers, progress):
        n_possible_grids = len(mesh_cell_centers)

        # 基于射线法，取+y方向的射线，判断和多边形的交点数
        if n_possible_grids > 0:
            edges = []
            n_edges = len(self.pts)
            for i in range(n_edges):
                edges.append(Edge(self.pts[i], self.pts[(i + 1) % n_edges]))

            mesh = mesh_cell_centers[:, 0:2]
            n_intersects = np.zeros(n_possible_grids, dtype=np.intp)

            idx = 0
            for edge in edges:
                intercepts = edge.intersects(mesh)
                n_intersects[:] += intercepts

                idx += 1
                if progress and idx % Prism.EdgesPerProgress == 0:
                    # 每若干条边更新一次进度条
                    progress.update(1)

            if progress and idx % Prism.EdgesPerProgress != 0:
                # 额外多出的边不足以更新进度条，在此处额外补足一次
                progress.update(1)

            indices_horizontally_satisfied = (n_intersects & 1) == 1
        else:
            indices_horizontally_satisfied = np.full(n_possible_grids, False)

        return indices_horizontally_satisfied

    def merge_transforms(self) -> Self:
        """将棱柱实例上附加的空间变换（translate、rotate等）应用到其二维轮廓点上

        Returns
        -------
        transformed_prism
            合并变换后的新棱柱实例，且不包含附加的空间变换

        Notes
        -----
        要求附加的空间变换不会改变坐标点的 z 轴坐标
        """
        z_plane_rotated_msg = (
            f'Rotation of z-plane detected.'
            f'Only transforms without rotating z-plane are supported for {type(self).__name__}.'
        )

        pts0 = np.c_[self.pts, np.full(len(self.pts), self.z0)]
        pts0_ = self.transforms.transform(pts0)

        assert np.allclose(pts0_[:, 2], pts0_[0, 2]), z_plane_rotated_msg

        pts1 = np.c_[self.pts[:2], np.full(2, self.z1)]
        pts1_ = self.transforms.transform(pts1)

        assert np.allclose(pts1_[:, 2], pts1_[0, 2]), z_plane_rotated_msg

        return Prism(pts=pts0_[:, :2], z0=pts0_[0, 2], z1=pts1_[0, 2])

    def do_compute_signed_distance(self, mesh_cell_centers: TransformedArray, progress):
        return self._do_compute_signed_distance(mesh_cell_centers, kernel=is_serial(), progress=progress)

    def _do_compute_signed_distance_taichi(self, mesh_cell_centers, progress):
        return self._do_compute_signed_distance(mesh_cell_centers, kernel=True, progress=progress)

    def _do_compute_signed_distance_numpy(self, mesh_cell_centers, progress):
        return self._do_compute_signed_distance(mesh_cell_centers, kernel=False, progress=progress)

    def _do_compute_signed_distance(self, mesh_cell_centers: TransformedArray, progress, kernel=True):
        # 将特殊角度值常量固定下来，方便后续判断
        angle_mapping = np.arctan2(
            [0, 1, 0, -1],
            [1, 0, -1, 0]
        )
        angle_constant = limit_significand(np.asarray([0, np.pi / 2, np.pi, -np.pi / 2]), tol=1e-5)

        pts = self.pts
        closed_pts = np.r_[pts, pts[:1]]
        line_ends = np.lib.stride_tricks.sliding_window_view(closed_pts, (2, 2)).squeeze()

        # 计算将每一条边线段平移到原点后旋转到与+x对齐的变换参数
        angles = []  # 线段关于+x方向的偏转角
        rot = []  # 线段偏转角的逆变换旋转矩阵 (2×2)
        intervals = []  # 线段旋转到与x轴对齐后的区间
        origins = []  # 线段的起点 (即线段的平移向量)
        for i, j in line_ends:
            edge = Edge(i, j)

            alpha = edge.signed_theta
            origin = edge.p1

            intervals.append([0, edge.length])
            origins.append(origin)
            rot.append([[np.cos(-alpha), -np.sin(-alpha)], [np.sin(-alpha), np.cos(-alpha)]])

            mapped = np.isclose(alpha, angle_mapping)
            if np.any(mapped):
                alpha = angle_constant[mapped].item()
            angles.append(alpha)

        angles = np.asarray(angles)
        rot = np.asarray(rot)
        origins = np.asarray(origins)
        intervals = np.asarray(intervals)

        inside2d = self.place2d(mesh_cell_centers, progress=False)

        dist = np.empty(mesh_cell_centers.shape[0], dtype=mesh_cell_centers.dtype)

        if kernel:
            func = _compute_signed_distance_element_wise.kernel
        else:
            func = _compute_signed_distance_element_wise

        func(
            mesh_cell_centers,
            inside2d=inside2d,
            angles=angles,
            origins=origins,
            rot=rot,
            intervals=intervals,
            z0=self.z0, z1=self.z1,
            c_px=angle_constant[0],
            c_py=angle_constant[1],
            c_nx=angle_constant[2],
            c_ny=angle_constant[3],
            dist=dist
        )

        if progress:
            progress.update(self.n_tasks)

        return dist

    def __dhash__(self):
        return dhash(
            super().__dhash__(),
            self.z0, self.z1,
            *self.pts.ravel(),
            self._cells if self._use_external_cells else None
        )

    def do_clone(self, deep=True):
        return Prism(self.pts.copy(), self.z0, self.z1)

    def _check_triangulated(self):
        """先检查 `pts` 是否属于简单多边形，如果为复杂多边形，则对内交点进行拆分与调整

        然后再进行三角化得到最终 `cells`
        """
        if self._pts is None:
            pts = self.pts
            if self.simplify:
                pts = re_intersect(pts, verbose=self.verbose)

            self._pts = pts
            self._cells = ear_clip(self._pts, verbose=self.verbose, check_non_simple=self.check_non_simple)

    @property
    def triangulated_points(self):
        self._check_triangulated()
        return self._pts

    @property
    def triangulated_cells(self):
        self._check_triangulated()
        return self._cells

    @triangulated_cells.setter
    def triangulated_cells(self, cells):
        """直接指定cells，跳过复杂多边形检查
        """
        cells_shape = (len(self.pts) - 2, 3)
        cells = np.asarray(cells)
        assert cells.shape == cells_shape, \
            f'`cells` must have shape {cells_shape},' \
            f' containing triangulated polygon cells. (got {cells.shape})'

        self._cells = np.asarray(cells)
        self._pts = self.pts
        self._use_external_cells = True

    @property
    def h(self):
        return self.z1 - self.z0

    @property
    def local_bounds(self):
        return Bounds(np.c_[
            np.r_[self.pts.min(axis=0), self.z0],
            np.r_[self.pts.max(axis=0), self.z1]
        ].ravel())

    def to_local_polydata(self):
        import pyvista as pv

        pts = self.triangulated_points
        z0, z1 = self.z0, self.z1
        n_pts = len(pts)
        n_vertices = 2 * len(pts)
        vh = pts.repeat(2, axis=0)
        vz = np.asarray([[z0, z1]]).repeat(n_pts, axis=0).ravel()
        vertices = np.c_[vh, vz]
        indices = np.arange(0, n_vertices, 1).reshape([-1, 2]).T

        faces = self.triangulated_cells
        top_face = np.c_[np.full(faces.shape[0], 3), faces * 2].ravel()  # 上顶面
        bottom_face = np.c_[np.full(faces.shape[0], 3), faces * 2 + 1].ravel()  # 下底面
        edge_counts = np.full(n_pts, 4)
        side_faces = np.c_[
            edge_counts,
            indices[0], indices[1],
            np.roll(indices[1], -1), np.roll(indices[0], -1)
        ].ravel()

        faces = np.r_[top_face, bottom_face, side_faces]

        shape = pv.PolyData(vertices.astype(np.float64), faces=faces)
        return shape

    @property
    def bottom_area(self):
        pts = self.triangulated_points
        s = 0
        for vi in self.triangulated_cells:
            v = pts[vi]
            si = np.abs(np.cross(v[1] - v[0], v[2] - v[0])) / 2
            s += si

        return s

    @property
    def volume(self):
        return self.bottom_area * self.h

    @property
    def area(self):
        ba = self.bottom_area
        rolled_pts = np.roll(self.pts, 1, axis=0)
        perimeter = np.linalg.norm(self.pts - rolled_pts, axis=1).sum()

        return 2 * ba + perimeter * self.h

    def progress_manually(self):
        return True

    @property
    def n_tasks(self):
        return np.ceil(len(self.pts) / Prism.EdgesPerProgress).astype(np.intp)


@ti_lazy_kernel
def _compute_signed_distance_element_wise(
        cell_centers: ti.types.ndarray(),
        inside2d: ti.types.ndarray(),
        angles: ti.types.ndarray(),
        origins: ti.types.ndarray(),
        rot: ti.types.ndarray(),
        intervals: ti.types.ndarray(),
        z0: ti.f64,
        z1: ti.f64,
        c_px: ti.f64,
        c_py: ti.f64,
        c_nx: ti.f64,
        c_ny: ti.f64,
        dist: ti.types.ndarray()
):
    n_cells = cell_centers.shape[0]
    n_edges = angles.shape[0]

    for i in range(n_cells):
        x, y, z = cell_centers[i, 0], cell_centers[i, 1], cell_centers[i, 2]
        dist2d = np.inf + x * 0  # 继承 cell_centers 的数据类型

        dist_z = min(abs(z - z1), abs(z - z0))
        is_inside2d = inside2d[i]
        is_inside_z = z0 <= z <= z1

        if is_inside2d and not is_inside_z:
            dist[i] = dist_z
        else:
            for j in range(n_edges):
                alpha = angles[j]
                fx, fy = origins[j, 0], origins[j, 1]
                tx, ty = x - fx, y - fy

                if alpha == c_px:
                    pass
                elif alpha == c_py:
                    tx, ty = ty, -tx
                elif alpha == c_nx:
                    tx, ty = -tx, -ty
                elif alpha == c_ny:
                    tx, ty = -ty, tx
                else:
                    tx, ty = rot[j, 0, 0] * tx + rot[j, 0, 1] * ty, rot[j, 1, 0] * tx + rot[j, 1, 1] * ty

                # 在边辐射范围内：取P到边的距离
                # 在边辐射范围外：取P到对应端点的点的距离
                # 结果即为所有边线段距离的最小值
                if tx > intervals[j, 1]:
                    pass  # 在辐射范围以右：不参与计算，由下一条边完成判断和计算
                else:
                    _dist2d = ty
                    if tx < intervals[j, 0]:
                        _dist2d = math.sqrt(tx * tx + ty * ty)  # 在辐射范围以左：计算P到左端点距离
                    else:
                        _dist2d = abs(ty)  # 在辐射范围以内：计算P到边距离

                    dist2d = min(dist2d, _dist2d)

            if is_inside2d:  # 此时必定 is_inside_z == True
                dist[i] = -min(dist2d, dist_z)
            elif is_inside_z:
                dist[i] = dist2d
            else:
                dist[i] = math.sqrt(dist_z ** 2 + dist2d ** 2)
