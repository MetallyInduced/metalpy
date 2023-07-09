import numpy as np

from metalpy.utils.dhash import dhash
from metalpy.utils.bounds import Bounds
from . import Shape3D


class Ellipsoid(Shape3D):
    def __init__(self, a, b, c):
        """定义一个椭球体，球心在原点

        Parameters
        ----------
        a
            椭球体在x方向上的半轴长
        b
            椭球体在y方向上的半轴长
        c
            椭球体在z方向上的半轴长
        """
        super().__init__()
        self.radii = np.asarray([a, b, c])

    @staticmethod
    def sphere(r):
        """定义一个球体

        Parameters
        ----------
        r
            半径

        Returns
        -------
        ret
            半径为r的球体
        """
        return Ellipsoid(r, r, r)

    @staticmethod
    def spheroid(a, c, polar_axis=2):
        """定义一个旋转椭球体（三个轴长为a, a, c）

        a > c时，其为扁椭球体

        a < c时，其为长椭球体

        默认polar_axis = 2时，其极轴在z轴上，方程为
            (x^2 + y^2) / a^2 + z^2 / c^2 = 1

        Parameters
        ----------
        a
            equatorial，半赤道轴长
        c
            polar，半极轴长
        polar_axis
            极轴方向，0 - x，1 - y，2 - z

        Returns
        -------
        ret
            中心点在原点，半赤道轴长为a，半极轴长为c，极轴为polar_axis的旋转椭球体
        """
        r = [a, a, a]
        r[polar_axis] = c
        return Ellipsoid(*r)

    def do_place(self, mesh_cell_centers, progress):
        return np.sum((mesh_cell_centers / self.radii) ** 2, axis=1) < 1

    def do_hash(self):
        return hash((*self.radii,))

    def __dhash__(self):
        return dhash(super().__dhash__(), *self.radii)

    def do_clone(self, deep=True):
        return Ellipsoid(*self.radii)

    @property
    def local_bounds(self):
        return Bounds(np.c_[-self.radii, self.radii].ravel())

    def to_local_polydata(self):
        import pyvista as pv

        return pv.ParametricEllipsoid(*self.radii)

    @property
    def volume(self):
        return np.pi * 4 / 3 * np.prod(self.radii)

    @property
    def area(self):
        return np.pi * 4 / 3 * (self.radii * np.roll(self.radii, 1)).sum()
