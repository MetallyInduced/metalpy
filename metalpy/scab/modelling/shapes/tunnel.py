import numpy as np

from metalpy.utils.dhash import dhash
from metalpy.utils.bounds import Bounds
from . import Shape3D


def is_abs_distance_in(arr, x0, r):
    return np.abs(arr - x0) < r


class Tunnel(Shape3D):
    def __init__(self, p0, r0, r1, L):
        """定义一个起始点在p0，内外圆半径r0和r1，向x方向延申长度为L的空心圆柱环

        Parameters
        ----------
        p0
            起始点，隧道的左中位置
        r0
            内圆半径
        r1
            外圆半径
        L
            隧道长度
        """
        super().__init__()
        self.p0 = np.asarray(p0)
        self.r0 = r0
        self.r1 = r1
        self.L = L

    def do_place(self, mesh_cell_centers, progress):
        x0, y0, z0 = self.p0
        L = self.L
        r0, r1 = self.r0, self.r1
        xs, ys, zs = mesh_cell_centers[:, 0], mesh_cell_centers[:, 1], mesh_cell_centers[:, 2]
        # 判定是否在长L，宽高r1的长方体范围内
        included = \
            is_abs_distance_in(xs, x0 + L / 2, L / 2) & \
            is_abs_distance_in(ys, y0, r1) & \
            is_abs_distance_in(zs, z0, r1)

        indices = included

        possible_grids = mesh_cell_centers[included, :]  # 只计算在这个长方体区域内的方块是否属于圆环柱
        rs = np.sqrt((possible_grids[:, 1] - self.p0[1]) ** 2 + (possible_grids[:, 2] - self.p0[2]) ** 2)
        indices[included] = (rs < r1) & (rs > r0)

        return indices

    def __dhash__(self):
        return dhash(super().__dhash__(),
                     self.r0, self.r1, self.L, *self.p0)

    def do_clone(self, deep=True):
        return Tunnel(
            self.p0.copy(),
            self.r0,
            self.r1,
            self.L
        )

    @property
    def local_bounds(self):
        x0, y0, z0 = self.p0
        r1 = self.r1
        L = self.L
        return Bounds(np.r_[
            x0, L,
            y0 - r1, y0 + r1,
            z0 - r1, z0 + r1,
        ].ravel())

    def to_local_polydata(self):
        import pyvista as pv

        resolution = 100
        a = np.linspace(0, 2 * np.pi, resolution + 1)[:-1]  # 2π位置和0位置重复，去掉最后一个
        x0, y0, z0 = self.p0
        r0, r1 = self.r0, self.r1
        L = self.L
        x = np.r_[x0, x0 + L]
        y = np.r_[y0 + r0 * np.cos(a), y0 + r1 * np.cos(a)]
        z = np.r_[z0 + r0 * np.sin(a), z0 + r1 * np.sin(a)]

        # array(resolution, ) [<x0内圆...>, <x0外圆...>, <x0+L内圆...>, <x0+L外圆...>]
        xs = x.repeat(2 * resolution)
        ys = np.tile(y, 2)
        zs = np.tile(z, 2)

        indices = np.arange(xs.shape[0]).reshape([4, -1])
        rolled = np.roll(indices, -1, axis=1)
        edge_counts = np.full(resolution, 3)

        face_vertices = [
            [0, 1],  # 底面
            [2, 3],  # 顶面
            [1, 3],  # 外圈表面
            [0, 2],  # 内圈表面
        ]

        faces = []
        for i0, i1 in face_vertices:
            faces.extend([  # 这个三角面
                edge_counts,
                indices[i0],
                indices[i1],
                rolled[i1]
            ])
            faces.extend([  # 那个三角面
                edge_counts,
                rolled[i1],
                rolled[i0],
                indices[i0]
            ])

        # 每次都 column_stac 然后 vstack 再 ravel
        # 和
        # 直接全部 column_stac 到一起再 ravel
        # 对PyVista而言等价
        faces = np.column_stack(faces).ravel()

        ret = pv.PolyData(np.c_[xs, ys, zs], faces=faces)
        ret.flip_normals()

        return ret

    @property
    def bottom_area(self):
        return np.pi * (self.r1 ** 2 - self.r0 ** 2)

    @property
    def volume(self):
        return self.bottom_area * self.L

    @property
    def area(self):
        ba = self.bottom_area
        p0 = 2 * np.pi * self.r0
        p1 = 2 * np.pi * self.r1

        return 2 * ba + (p0 + p1) * self.L
