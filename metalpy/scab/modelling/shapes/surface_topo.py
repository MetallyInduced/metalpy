import sys

import numpy as np

from metalpy.utils.bounds import Bounds
from metalpy.utils.dhash import dhash
from metalpy.utils.string import parse_axes_labels, parse_labels
from . import InfiniteShape


class SurfaceTopo(InfiniteShape):
    def __init__(self, loc, alt=None, loc_axes=None, alt_axis=None, inverse=False, sample=None):
        """定义一个地形几何体

        Parameters
        ----------
        loc
            地形位置坐标
        alt
            地形高度，若未指定，则取`loc`的最后一列为高度
        loc_axes
            地形位置所属的坐标轴，默认为 `xy`
        alt_axis
            地形的外法线方向，指向地面以上，例如
            `+z` / `z` / `2` 代表垂直xy平面向+z方向，则-z方向部分为 `地下`，
            `-y` / `-1` 代表垂直xz平面向-y方向，则+y方向部分为 `地下`
        inverse
            翻转地形的外法线方向。
            注：如果 `alt_axis` 为 `-z` 则会变为 `+z`
        sample
            对曲面上的点进行采样的采样间距，等价于 loc[::sample], alt[::sample]
        """
        super().__init__()

        if loc_axes is not None:
            loc_axes = parse_axes_labels(loc_axes, max_length=2)
            loc_axes = list(set(loc_axes))

        if alt_axis is not None:
            self.alt_axis, self.inverse = parse_axis_name(alt_axis)
            if loc_axes is None:
                loc_axes = list({0, 1, 2}.difference(self.alt_axis))
            self.loc_axes = loc_axes
        elif loc_axes is not None:
            self.loc_axes = loc_axes
            assert self.surface_dim == 2, 'Only 3D surface can have `alt_axis` inferred from `loc_axes`.'
            self.alt_axis, self.inverse = outer_product(loc_axes)
        else:
            raise RuntimeError(
                'Either of `loc_axes` or `alt_axis` must be provided.'
            )

        if inverse:
            self.inverse = not self.inverse

        loc = np.asarray(loc)
        if alt is None:
            alt = loc[:, self.alt_axis]
            loc = loc[:, self.loc_axes]
        else:
            alt = np.asarray(alt)
            assert len(loc) == len(alt), '`loc` and `alt` must have same length.'

        assert self.surface_dim == loc.shape[1], '`loc_axes` must have same dimensions with locations.'

        if sample is not None:
            loc = loc[::sample]
            alt = alt[::sample]

        self.loc = loc
        self.alt = alt

    @property
    def surface_dim(self):
        return len(self.loc_axes)

    @property
    def n_pts(self):
        return len(self.loc)

    @property
    def n_dims(self):
        return self.surface_dim + 1

    def track(self, points):
        from scipy.interpolate import LinearNDInterpolator

        fun_interp = LinearNDInterpolator(self.loc, self.alt)
        return fun_interp(points)

    @staticmethod
    def xy2z(loc, alt=None, inverse=False, sample=None):
        return SurfaceTopo.ab2c(loc, alt=alt, loc_axes='xy', alt_axis='z', inverse=inverse, sample=sample)

    @staticmethod
    def yz2x(loc, alt=None, inverse=False, sample=None):
        return SurfaceTopo.ab2c(loc, alt=alt, loc_axes='yz', alt_axis='x', inverse=inverse, sample=sample)

    @staticmethod
    def zx2y(loc, alt=None, inverse=False, sample=None):
        return SurfaceTopo.ab2c(loc, alt=alt, loc_axes='zx', alt_axis='y', inverse=inverse, sample=sample)

    @staticmethod
    def x2y(loc, alt=None, inverse=False, sample=None):
        return SurfaceTopo.ab2c(loc, alt=alt, loc_axes='x', alt_axis='y', inverse=inverse, sample=sample)

    @staticmethod
    def y2z(loc, alt=None, inverse=False, sample=None):
        return SurfaceTopo.ab2c(loc, alt=alt, loc_axes='y', alt_axis='z', inverse=inverse, sample=sample)

    @staticmethod
    def z2x(loc, alt=None, inverse=False, sample=None):
        return SurfaceTopo.ab2c(loc, alt=alt, loc_axes='z', alt_axis='x', inverse=inverse, sample=sample)

    @staticmethod
    def ab2c(loc, alt=None, loc_axes='xy', alt_axis='z', inverse=False, sample=None):
        pv = sys.modules.get('pyvista', None)
        if pv is not None:
            if isinstance(loc, pv.DataSet):
                return SurfaceTopo.from_polydata(
                    loc,
                    loc_axes=loc_axes,
                    alt_axis=alt_axis,
                    inverse=inverse,
                    sample=sample
                )
        return SurfaceTopo(
            loc, alt=alt,
            loc_axes=loc_axes,
            alt_axis=alt_axis,
            inverse=inverse,
            sample=sample
        )

    @staticmethod
    def from_polydata(poly, loc_axes=None, alt_axis=None, inverse=False, sample=None):
        from metalpy.utils.pyvista_dataset_wrappers.universal_dataset import UniversalDataSet
        points = UniversalDataSet(poly).points

        if loc_axes is None and alt_axis is None:
            maxs, mins = points.max(axis=0), points.min(axis=0)
            scaled = (points - mins) / (maxs - mins)
            # 协方差最接近0，代表其在对应平面上分布最均匀，因此大概率为坐标平面
            cov = np.cov(scaled, rowvar=False).ravel()[[1, 2, 5]]
            loc_axes = [[0, 1], [1, 2], [2, 0]][np.argmin(np.abs(cov))]

        return SurfaceTopo(
            points,
            loc_axes=loc_axes,
            alt_axis=alt_axis,
            inverse=inverse,
            sample=sample
        )

    def do_place(self, mesh_cell_centers, progress):
        cell_locs = mesh_cell_centers[:, self.loc_axes]
        cell_alts = mesh_cell_centers[:, self.alt_axis]

        if not self.inverse:
            return cell_alts < self.track(cell_locs)
        else:
            return cell_alts > self.track(cell_locs)

    def __dhash__(self):
        return dhash(
            super().__dhash__(),
            self.loc,
            self.alt,
            self.loc_axes,
            self.alt_axis,
            self.inverse,
        )

    def do_clone(self, deep=True):
        return SurfaceTopo(loc=self.loc, alt=self.alt, loc_axes=self.loc_axes, alt_axis=self.alt_axis,
                           inverse=self.inverse)

    @property
    def local_bounds(self):
        ret = Bounds.unbounded(self.n_dims)

        for idx, a in enumerate(self.loc_axes):
            c = self.loc[:, idx]
            ret.set(a, min=c.min(), max=c.max())

        if self.inverse:
            ret.set(self.alt_axis, min=self.alt.min())
        else:
            ret.set(self.alt_axis, max=self.alt.max())

        return ret

    def to_local_polydata(self):
        import pyvista as pv

        if self.surface_dim < 2:
            return None  # TODO: Scene添加接口支持获取场景信息来构造polydata

        pts = np.empty(
            (self.n_pts, self.surface_dim + 1),
            dtype=np.result_type(self.loc, self.alt)
        )
        pts[:, self.loc_axes] = self.loc
        pts[:, self.alt_axis] = self.alt

        from scipy.spatial import Delaunay
        delaunay = Delaunay(self.loc)
        faces = delaunay.simplices
        _3 = np.full(len(faces), 3)

        return pv.PolyData(pts, np.c_[_3, faces].ravel())


def axis_direction(axis, inverse=False):
    if not inverse:
        return f'+{axis}'
    else:
        return f'-{axis}'


def parse_axis_name(axis):
    if isinstance(axis, int):
        if axis > 0:
            inverse = False
        else:
            inverse = True
        axis = abs(axis)
    elif len(axis) == 1:
        inverse = False
    elif len(axis) == 2:
        inverse = [False, True][parse_labels(axis[0], accepts='+-', length=1)[0]]
        axis = axis[1]
    else:
        raise RuntimeError(
            f'`axis` must be axis name like `+x`, `-y` or `z`(alias to `+z`),'
            f' got {axis}.'
        )
    axis = parse_axes_labels(axis, length=1)[0]

    return axis, inverse


def outer_product(axes):
    """根据右手定则确定第三轴方向，认为 `axes` 分别指定了两个坐标轴的正方向

    则 `xy` , `yz` , `zx` 分别对应 `+z` , `+x` , `+y`

    Parameters
    ----------
    axes
        参与计算的两个坐标轴，必须为 0, 1, 2 中的两个

    Returns
    -------
    axis_with_sign
        返回外积下的第三轴方向
    """
    axes_set = set(axes)
    assert len(axes_set) == 2 and len(axes_set.difference({0, 1, 2})) == 0

    axis = list({0, 1, 2}.difference(axes))[0]
    inverse = (axes[1] - axes[0]) % 3 == 2

    return axis, inverse
