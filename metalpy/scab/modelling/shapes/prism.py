from __future__ import annotations

import math
from typing import TypeVar

import numpy as np
from scipy.stats import linregress

from metalpy.mepa.utils import is_serial
from metalpy.utils.bounds import Bounds
from metalpy.utils.dhash import dhash
from metalpy.utils.ear_clip import ear_clip
from metalpy.utils.geometry import gen_random_convex_polygon
from metalpy.utils.numeric import limit_significand
from metalpy.utils.rand import check_random_state
from metalpy.utils.ti_lazy import ti_lazy_kernel, ti
from . import Shape3D
from .shape3d import TransformedArray

PrismT = TypeVar('PrismT', bound='Prism')


def is_abs_distance_in(arr, x0, r):
    return np.abs(arr - x0) < r


def is_in_multipoints_or_point(pt, mp_or_pt):
    if mp_or_pt.geom_type == 'Point':
        return pt == mp_or_pt
    else:
        return pt in mp_or_pt.geoms


class Edge:
    def __init__(self, p1, p2):
        self.p1, self.p2 = p1, p2
        self.ymin = min(p1[1], p2[1])
        self.ymax = max(p1[1], p2[1])
        self.xmin = min(p1[0], p2[0])
        self.xmax = max(p1[0], p2[0])

        ends = np.vstack([p1, p2])

        if p1[0] != p2[0]:  # 非垂直于x轴
            slope, intercept, _, _, _ = linregress(ends[:, 0], ends[:, 1])
            self.k, self.b = slope, intercept
        else:
            self.k = np.inf
            self.x0 = p1[0]

    @property
    def signed_theta(self):
        return np.arctan2(*(self.p2 - self.p1)[::-1])

    @property
    def length(self):
        return np.linalg.norm(self.p2 - self.p1)

    def intersects(self, pts):
        x, y = pts[:, 0], pts[:, 1]
        if np.isinf(self.k):
            # 处理与垂直线相交的情形
            # 由于是多面体，因此与该直线相交的射线一定与其重合
            # 其所在直线会与该边的相邻两个边也存在交点
            # 交于下端点视为相交，其余位置不视为相交
            # 可以保证边缘点被视为多边形内
            # 1. 点p在该线段之下，则只会与相邻边有2个交点
            # 2. 点p在该线段与下方邻边交点，则会与相邻边和该边有3个交点
            # 3. 点p在该线段中间及上方邻边交点，则只会与上方邻边有1个交点
            indices = x == self.x0
            indices[indices] = y[indices] == self.ymin
            return indices
        elif self.k == 0:
            return (x >= self.xmin) & (x <= self.xmax) & (y <= self.ymin)
        else:
            y_intersection = self.k * x + self.b
            return (y_intersection >= self.ymin) & (y_intersection <= self.ymax) & (y <= y_intersection)


class Prism(Shape3D):
    def __init__(self,
                 pts,
                 z0: float | int = 0,
                 z1: float | int = 1,
                 cells=None,
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
        check_non_simple
            是否检查非简单多边形。为 `True` 时遇到非简单多边形会抛出异常，否则会继续执行（但大概率结果错误）
        verbose
            是否输出辅助信息（主要为顶点数较多时导出PyVista模型会触发进度条）
        """
        super().__init__()
        self.pts = np.asarray(pts)
        self.z0 = min(z0, z1)
        self.z1 = max(z0, z1)

        if cells is not None:
            cells_shape = (len(self.pts) - 2, 3)
            cells = np.asarray(cells)
            assert cells.shape == cells_shape, \
                f'`cells` must have shape {cells_shape},' \
                f' containing triangulated polygon cells. (got {cells.shape})'
            self.triangulated_polygon = np.asarray(cells)
        else:
            self.triangulated_polygon = ear_clip(self.pts, verbose=verbose, check_non_simple=check_non_simple)

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

    @property
    def h(self):
        return self.z1 - self.z0

    def do_place(self, mesh_cell_centers, progress):
        n_possible_grids = len(mesh_cell_centers)

        # 基于射线法，取+y方向的射线，判断和多边形的交点数
        if n_possible_grids > 0:
            edges = []
            n_edges = len(self.pts)
            for i in range(n_edges):
                edges.append(Edge(self.pts[i], self.pts[(i + 1) % n_edges]))

            mesh = mesh_cell_centers[:, 0:2]
            n_intersects = np.zeros(n_possible_grids, dtype=int)

            for edge in edges:
                intercepts = edge.intersects(mesh)
                n_intersects = n_intersects + intercepts

            indices_horizontally_satisfied = (n_intersects & 1) == 1
        else:
            indices_horizontally_satisfied = np.full(n_possible_grids, False)

        return indices_horizontally_satisfied

    def do_compute_signed_distance(self, mesh_cell_centers: TransformedArray, progress):
        return self._do_compute_signed_distance(mesh_cell_centers, kernel=is_serial())

    def _do_compute_signed_distance_taichi(self, mesh_cell_centers):
        return self._do_compute_signed_distance(mesh_cell_centers, kernel=True)

    def _do_compute_signed_distance_numpy(self, mesh_cell_centers):
        return self._do_compute_signed_distance(mesh_cell_centers, kernel=False)

    def _do_compute_signed_distance(self, mesh_cell_centers: TransformedArray, kernel=True):
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

        return dist

    def __dhash__(self):
        return dhash(
            super().__dhash__(),
            self.z0, self.z1,
            *self.pts.ravel(),
            self.triangulated_polygon
        )

    def do_clone(self, deep=True):
        return Prism(self.pts.copy(), self.z0, self.z1)

    @property
    def local_bounds(self):
        return Bounds(np.c_[
            np.r_[self.pts.min(axis=0), self.z0],
            np.r_[self.pts.max(axis=0), self.z1]
        ].ravel())

    def to_local_polydata(self):
        import pyvista as pv

        pts = self.pts
        z0, z1 = self.z0, self.z1
        n_pts = len(pts)
        n_vertices = 2 * len(pts)
        vh = pts.repeat(2, axis=0)
        vz = np.asarray([[z0, z1]]).repeat(n_pts, axis=0).ravel()
        vertices = np.c_[vh, vz]
        indices = np.arange(0, n_vertices, 1).reshape([-1, 2]).T

        faces = self.triangulated_polygon
        top_face = np.c_[np.repeat(3, faces.shape[0]), faces * 2].ravel()  # 上顶面
        bottom_face = np.c_[np.repeat(3, faces.shape[0]), faces * 2 + 1].ravel()  # 下底面
        edge_counts = np.ones(n_pts, dtype=np.integer) * 4
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
        s = 0
        for vi in self.triangulated_polygon:
            v = self.pts[vi]
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
