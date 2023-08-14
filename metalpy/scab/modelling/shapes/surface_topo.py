import numpy as np

from metalpy.utils.bounds import Bounds
from metalpy.utils.dhash import dhash
from metalpy.utils.string import parse_axes_labels
from . import InfiniteShape


class SurfaceTopo(InfiniteShape):
    def __init__(self, loc, alt=None, loc_axes='xy', alt_axis=None, upwards=False):
        """定义一个地形几何体

        Parameters
        ----------
        loc
            地形位置坐标
        alt
            地形高度，若未指定，则取`loc`的最后一列为高度
        loc_axes
            地形位置所属的坐标轴
        alt_axis
            地形高度所属坐标轴
        upwards
            地形向上延申，默认为False，即向 -alt_axis 方向延申
        """
        super().__init__()

        loc = np.asarray(loc)
        if alt is None:
            alt = loc[:, -1]
            loc = loc[:, :-1]
        else:
            alt = np.asarray(alt)

        self.loc = loc
        self.alt = alt
        assert len(loc) == len(alt), '`loc` and `alt` must have same length.'

        self.loc_axes = parse_axes_labels(loc_axes, max_length=2)
        assert self.surface_dim == loc.shape[1], '`loc_axes` must have same dimensions with locations.'

        if alt_axis is not None:
            self.alt_axis = parse_axes_labels(alt_axis, length=1)[0]
        else:
            assert self.surface_dim == 2, 'Only 2D surface can have `alt_axis` inferred from `loc_axes`.'
            self.alt_axis = list({0, 1, 2}.difference(self.loc_axes))[0]

        self.upwards = upwards

    @staticmethod
    def xy2z(loc, alt=None, upwards=False):
        return SurfaceTopo(loc, alt=alt, loc_axes='xy', alt_axis='z', upwards=upwards)

    @staticmethod
    def yz2x(loc, alt=None, upwards=False):
        return SurfaceTopo(loc, alt=alt, loc_axes='yz', alt_axis='x', upwards=upwards)

    @staticmethod
    def zx2y(loc, alt=None, upwards=False):
        return SurfaceTopo(loc, alt=alt, loc_axes='zx', alt_axis='y', upwards=upwards)

    @staticmethod
    def x2y(loc, alt=None, upwards=False):
        return SurfaceTopo(loc, alt=alt, loc_axes='x', alt_axis='y', upwards=upwards)

    @staticmethod
    def y2z(loc, alt=None, upwards=False):
        return SurfaceTopo(loc, alt=alt, loc_axes='y', alt_axis='z', upwards=upwards)

    @staticmethod
    def z2x(loc, alt=None, upwards=False):
        return SurfaceTopo(loc, alt=alt, loc_axes='z', alt_axis='x', upwards=upwards)

    @property
    def surface_dim(self):
        return len(self.loc_axes)

    @property
    def n_pts(self):
        return len(self.loc)

    @property
    def n_dims(self):
        return self.surface_dim + 1

    def do_place(self, mesh_cell_centers, progress):
        from scipy.interpolate import LinearNDInterpolator

        fun_interp = LinearNDInterpolator(self.loc, self.alt)
        cell_locs = mesh_cell_centers[:, self.loc_axes]
        cell_alts = mesh_cell_centers[:, self.alt_axis]

        if not self.upwards:
            return cell_alts < fun_interp(cell_locs)
        else:
            return cell_alts > fun_interp(cell_locs)

    def __dhash__(self):
        return dhash(
            super().__dhash__(),
            self.loc,
            self.alt,
            self.loc_axes,
            self.alt_axis,
            self.upwards,
        )

    def do_clone(self, deep=True):
        return SurfaceTopo(loc=self.loc, alt=self.alt, loc_axes=self.loc_axes, alt_axis=self.alt_axis,
                           upwards=self.upwards)

    @property
    def local_bounds(self):
        ret = Bounds.unbounded(self.n_dims)

        for idx, a in enumerate(self.loc_axes):
            c = self.loc[:, idx]
            ret.set(a, min=c.min(), max=c.max())

        if self.upwards:
            ret.set(self.alt_axis, min=self.alt.min())
        else:
            ret.set(self.alt_axis, max=self.alt.max())

        return ret

    def to_local_polydata(self):
        import pyvista as pv

        if self.surface_dim < 2:
            return None  # TODO: Scene添加接口支持获取场景信息来构造polydata

        pts = np.zeros((self.n_pts, 3), dtype=float)
        pts[:, self.loc_axes] = self.loc
        pts[:, self.alt_axis] = self.alt

        return pv.PolyData(pts).reconstruct_surface()
