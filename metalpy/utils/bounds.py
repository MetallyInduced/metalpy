from typing import cast

import numpy as np

from metalpy.utils.numpy import FixedShapeNDArray


def _regulate_nans(val, b1, b2):
    """b1 is asserted to be shorter than b2
    """
    length = b1.n_axes * 2

    mask = np.isnan(b1)[:length]
    val[:length][mask] = b2[:length][mask]

    mask = np.isnan(b2)[:length]
    val[:length][mask] = b1[:length][mask]


def _aligned_max(b1: 'Bounds', b2: 'Bounds'):
    """b1 is asserted to be shorter than b2
    """
    length = b1.n_axes * 2

    bounds = cast(
        Bounds,
        np.array(b2, dtype=np.promote_types(b1.dtype, b2.dtype), copy=True, subok=True)
    )
    bounds[:length] = np.max([b1[:length], b2[:length]], axis=0)

    return bounds, b1.n_axes


def union(b1: 'Bounds', b2: 'Bounds'):
    b1, b2 = (b2, b1) if b1.n_axes > b2.n_axes else (b1, b2)
    bounds, n_axes = _aligned_max(b1, b2)
    bounds.origin[:n_axes] = np.min([
        b1.origin[:n_axes],
        b2.origin[:n_axes]
    ], axis=0)
    _regulate_nans(bounds, b1, b2)
    return bounds


def intersects(b1: 'Bounds', b2: 'Bounds'):
    b1, b2 = (b2, b1) if b1.n_axes > b2.n_axes else (b1, b2)
    bounds, n_axes = _aligned_max(b1, b2)
    bounds.end[:n_axes] = np.min([
        b1.end[:n_axes],
        b2.end[:n_axes]
    ], axis=0)
    _regulate_nans(bounds, b1, b2)
    return bounds


def bounded(xmin=np.nan, xmax=np.nan, ymin=np.nan, ymax=np.nan, zmin=np.nan, zmax=np.nan, *, n_axes=None):
    return Bounds.partial(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        n_axes=n_axes
    )


class Bounds(FixedShapeNDArray):
    def __new__(cls, var_inp, *args, dtype=None) -> 'Bounds':
        if np.isscalar(var_inp):
            return np.asanyarray([var_inp, *args], dtype=dtype).view(cls)
        else:
            return np.asanyarray(var_inp, dtype=dtype).view(cls)

    def __array_finalize__(self, _, **__):
        pass

    @staticmethod
    def unbounded(n_axes=0) -> 'Bounds':
        return np.full(n_axes * 2, np.nan).view(Bounds)

    @staticmethod
    def partial(xmin=np.nan, xmax=np.nan, ymin=np.nan, ymax=np.nan, zmin=np.nan, zmax=np.nan, *, n_axes=None):
        if n_axes is None:
            n_axes = np.where(~np.isnan([0, 0, xmin, xmax, ymin, ymax, zmin, zmax]))[0][-1] // 2

        ret = Bounds.unbounded(n_axes)

        ret.xmin = xmin
        ret.xmax = xmax
        if n_axes > 1:
            ret.ymin = ymin
            ret.ymax = ymax
        if n_axes > 2:
            ret.zmin = zmin
            ret.zmax = zmax

        return ret

    def as_corners(self):
        """从边界形式转换为角落点形式

        Returns
        -------
        ret
            角落点形式表示的
        """
        return self.reshape(-1, 2).T.view(Corners)

    def expand(self, *, proportion=None, increment=None, inplace=False):
        number = (int, float)
        deltas = np.tile(np.asarray((-1, 1)), self.n_axes)
        if increment is None:
            assert proportion is not None, 'Either `proportion` or `increment` must be provided.'
            if isinstance(proportion, number):
                proportion = deltas * proportion
            increment = self.extent.repeat(2) * proportion
        elif isinstance(increment, number):
            increment = increment * deltas

        if inplace:
            target = self
        else:
            target = self.copy()

        target += increment
        return target

    def override(self, by, *, inplace=False):
        """用other中的非nan值替换当前边界中的对应位置值

        Parameters
        ----------
        by
            用于覆盖self的边界值
        inplace
            指示操作是否修改self

        Returns
        -------
        ret
            覆盖后的边界对象
        """
        target = self
        if not inplace:
            target = target.copy()

        other = Bounds(by)
        length = min(target.n_axes, other.n_axes) * 2
        mask = ~np.isnan(other)[:length]
        target[:length][mask] = other[:length][mask]

        return target

    def set(self, axis, min=None, max=None):
        if min is not None:
            self[2 * axis] = min

        if max is not None:
            self[2 * axis + 1] = max

    __or__ = union
    __and__ = intersects

    @property
    def extent(self):
        return self.end - self.origin

    @property
    def origin(self):
        return self[::2]

    @origin.setter
    def origin(self, val):
        self[::2] = val

    @property
    def end(self):
        return self[1::2]

    @end.setter
    def end(self, val):
        self[1::2] = val

    @property
    def xmin(self): return self[0]

    @xmin.setter
    def xmin(self, v): self[0] = v

    @property
    def xmax(self): return self[1]

    @xmax.setter
    def xmax(self, v): self[1] = v

    @property
    def xrange(self): return self[0:2]

    @xrange.setter
    def xrange(self, rng): self[0:2] = rng

    @property
    def ymin(self): return self[2]

    @ymin.setter
    def ymin(self, v): self[2] = v

    @property
    def ymax(self): return self[3]

    @ymax.setter
    def ymax(self, v): self[3] = v

    @property
    def yrange(self): return self[2:4]

    @yrange.setter
    def yrange(self, rng): self[2:4] = rng

    @property
    def zmin(self): return self[4]

    @zmin.setter
    def zmin(self, v): self[4] = v

    @property
    def zmax(self): return self[5]

    @zmax.setter
    def zmax(self, v): self[5] = v

    @property
    def zrange(self): return self[4:6]

    @zrange.setter
    def zrange(self, rng): self[4:6] = rng

    @property
    def n_axes(self): return self.shape[0] // 2


class Corners(FixedShapeNDArray):
    def __new__(cls, input_arr):
        return np.asanyarray(input_arr).view(cls)

    def as_bounds(self):
        """从角落点形式转换为边界形式

        Returns
        -------
        ret
            角边界形式表示的
        """
        return self.T.ravel().view(Bounds)

    def set(self, axis, min=None, max=None):
        if min is not None:
            self[0, axis] = min

        if max is not None:
            self[1, axis] = max

    @property
    def extent(self):
        return self.end - self.origin

    @property
    def origin(self):
        return self[0]

    @origin.setter
    def origin(self, val):
        self[0] = val

    @property
    def end(self):
        return self[1]

    @end.setter
    def end(self, val):
        self[1] = val

    @property
    def xmin(self): return self[0, 0]

    @xmin.setter
    def xmin(self, v): self[0, 0] = v

    @property
    def xmax(self): return self[1, 0]

    @xmax.setter
    def xmax(self, v): self[1, 0] = v

    @property
    def xrange(self): return self[:, 0]

    @xrange.setter
    def xrange(self, rng): self[:, 0] = rng

    @property
    def ymin(self): return self[0, 1]

    @ymin.setter
    def ymin(self, v): self[0, 1] = v

    @property
    def ymax(self): return self[1, 1]

    @ymax.setter
    def ymax(self, v): self[1, 1] = v

    @property
    def yrange(self): return self[:, 1]

    @yrange.setter
    def yrange(self, rng): self[:, 1] = rng

    @property
    def zmin(self): return self[0, 2]

    @zmin.setter
    def zmin(self, v): self[0, 2] = v

    @property
    def zmax(self): return self[1, 2]

    @zmax.setter
    def zmax(self, v): self[1, 2] = v

    @property
    def zrange(self): return self[:, 2]

    @zrange.setter
    def zrange(self, rng): self[:, 2] = rng

    @property
    def n_axes(self): return self.origin.shape[0]
