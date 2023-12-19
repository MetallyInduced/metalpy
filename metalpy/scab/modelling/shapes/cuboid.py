import math
from typing import Iterable, Union

import numpy as np

from metalpy.mepa.utils import is_serial
from metalpy.utils.bounds import Bounds
from metalpy.utils.dhash import dhash
from metalpy.utils.ti_lazy import ti_lazy_kernel, ti
from . import Shape3D


class Cuboid(Shape3D):
    def __init__(self,
                 origin: Iterable = None,
                 end: Iterable = None,
                 corner: Iterable = None,
                 corner2: Iterable = None,
                 size: Union[Iterable, float] = None,
                 center: Iterable = None,
                 bounds: Iterable = None,
                 no_corner_adjust=False):
        """通过立方体的中心、长度、角点中任意两个参数或单独使用边界来定义立方体

        Parameters
        ----------
        origin, end
            立方体的角点坐标
        corner, corner2
            立方体的角点坐标，用于兼容，建议使用origin和end
        size
            立方体的长宽高
        center
            中心点坐标
        bounds
            立方体的边界，当定义时，无视其他参数，[x0, x1, y0, y1, z0, z1]
        no_corner_adjust
            是否拒绝调整角点坐标，默认为假，出现负数边长时，会自动调整角点坐标，若为真，则抛出异常
        """
        super().__init__()

        if origin is None:
            origin = corner

        if end is None:
            end = corner2

        if bounds is None and np.r_[
            origin is not None,
            end is not None,
            center is not None,
            size is not None,
        ].sum() != 2:
            raise ValueError("Bounds or exactly two of"
                             " `center`, `lengths`, `origin`, `end`"
                             " must be specified.")

        if origin is not None:
            origin = np.array(origin)

        if size is not None:
            size = np.array(size)
            if size.size == 1:
                size = np.ones(3) * size

        if bounds is not None:
            origin = np.array(bounds[::2])
            size = np.array(bounds[1::2]) - origin
        elif origin is not None:  # 和end或center，计算size
            if end is not None:
                size = np.asarray(end) - np.asarray(origin)
            elif center is not None:
                size = 2 * (np.asarray(center) - np.asarray(origin))
        elif end is not None:  # 和center或size，计算origin
            if center is not None:
                size = 2 * (np.asarray(end) - np.asarray(center))
                origin = np.asarray(end) - size
            elif size is not None:
                origin = np.asarray(end) - np.asarray(size)
        elif size is not None:  # 和center，计算origin
            if center is not None:
                origin = np.asarray(center) - size / 2

        if np.any(size < 0):
            if no_corner_adjust:
                raise ValueError("Negative length detected.")
            # 纠正负数边长
            origin[size < 0] += size[size < 0]
            size = np.abs(size)

        self.origin = origin
        self.lengths = size

    def to_prism(self):
        from .prism import Prism
        return Prism(pts=self.pts, z0=self.z0, z1=self.z1)

    def do_place(self, mesh_cell_centers, progress):
        return np.full(len(mesh_cell_centers), True)

    def do_compute_signed_distance(self, mesh_cell_centers, progress):
        if is_serial():
            return self._do_compute_signed_distance_taichi(mesh_cell_centers)
        else:
            return self._do_compute_signed_distance_numpy(mesh_cell_centers)

    @property
    def x0(self): return self.origin[0]

    @property
    def x1(self): return self.origin[0] + self.lengths[0]

    @property
    def y0(self): return self.origin[1]

    @property
    def y1(self): return self.origin[1] + self.lengths[1]

    @property
    def z0(self): return self.origin[2]

    @property
    def z1(self): return self.origin[2] + self.lengths[2]

    @x0.setter
    def x0(self, val):
        self.lengths[0] = self.x1 - val
        self.origin[0] = val

    @x1.setter
    def x1(self, val): self.lengths[0] = val - self.x0

    @y0.setter
    def y0(self, val):
        self.lengths[1] = self.y1 - val
        self.origin[1] = val

    @y1.setter
    def y1(self, val): self.lengths[1] = val - self.y0

    @z0.setter
    def z0(self, val):
        self.lengths[2] = self.z1 - val
        self.origin[2] = val

    @z1.setter
    def z1(self, val): self.lengths[2] = val - self.z0

    @property
    def corner(self): return self.origin

    @property
    def end(self): return self.origin + self.lengths

    @property
    def pts(self):
        """兼容Prism接口"""
        return np.asarray([
            [self.x0, self.y0],
            [self.x0, self.y1],
            [self.x1, self.y1],
            [self.x1, self.y0],
        ])

    @property
    def direction(self):
        """
        获取横梁最长轴方向

        Returns
        -------
        axis
            0, 1, 2代表x, y, z
        """
        return np.argmax(self.lengths)

    def __dhash__(self):
        return dhash(super().__dhash__(), *self.origin, *self.lengths)

    def do_clone(self, deep=True):
        return Cuboid(
            origin=self.origin.copy(),
            size=self.lengths.copy()
        )

    @property
    def local_bounds(self):
        return Bounds(np.c_[self.origin, self.origin + self.lengths].ravel())

    def to_local_polydata(self):
        import pyvista as pv

        return pv.Cube(bounds=np.asarray(self.local_bounds))

    @property
    def volume(self):
        return np.prod(self.lengths)

    @property
    def area(self):
        return 2 * (self.lengths * np.roll(self.lengths, 1)).sum()

    def _do_compute_signed_distance_taichi(self, mesh_cell_centers):
        dist = np.empty(mesh_cell_centers.shape[0], dtype=mesh_cell_centers.dtype)
        _compute_signed_distance_element_wise.kernel(mesh_cell_centers, self.origin, self.end, dist=dist)

        return dist

    def _do_compute_signed_distance_numpy(self, mesh_cell_centers):
        d0 = self.origin - mesh_cell_centers
        d1 = self.end - mesh_cell_centers
        outer_dist = np.min([abs(d0), abs(d1)], axis=0)

        mask = (d0 <= 0) & (d1 >= 0)
        state = np.sum(mask, axis=1)
        inside = state == 3  # 在长方体内，求最近距离
        inside_2d = state == 2  # 在长方体的任一面外（上下左右前后），求第三轴垂直最近距离
        others = state < 2  # 在长方体的角落区域（前右、上左等角落区域），求到角点或交线的欧氏距离

        dist = np.empty_like(inside, dtype=d0.dtype)
        dist[inside_2d] = outer_dist[~mask * inside_2d[:, None]]
        dist[others] = np.linalg.norm(outer_dist[others] * (~mask[others]), axis=1)
        dist[inside] = -np.min(outer_dist[inside], axis=1)

        return dist


@ti_lazy_kernel
def _compute_signed_distance_element_wise(
        cells_centers: ti.types.ndarray(),
        origin: ti.types.ndarray(),
        end: ti.types.ndarray(),
        dist: ti.types.ndarray()
):
    for i in range(dist.shape[0]):
        dx0, dx1 = origin[0] - cells_centers[i, 0], end[0] - cells_centers[i, 0]
        dy0, dy1 = origin[1] - cells_centers[i, 1], end[1] - cells_centers[i, 1]
        dz0, dz1 = origin[2] - cells_centers[i, 2], end[2] - cells_centers[i, 2]

        cx: ti.i8 = dx0 <= 0 <= dx1
        cy: ti.i8 = dy0 <= 0 <= dy1
        cz: ti.i8 = dz0 <= 0 <= dz1

        state = cx + cy + cz

        dx = abs(min(-dx0, dx1))
        dy = abs(min(-dy0, dy1))
        dz = abs(min(-dz0, dz1))

        if state == 3:
            dist[i] = -min(dx, dy, dz)
        else:
            dx *= not cx
            dy *= not cy
            dz *= not cz
            if state == 2:
                dist[i] = dx + dy + dz
            else:
                dist[i] = math.sqrt(dx * dx + dy * dy + dz * dz)
