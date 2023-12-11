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


class Bounds(FixedShapeNDArray):
    """用于表示一个边界，接受数字和np.nan作为内容（不允许存在np.inf）

    使用np.nan用于表示在该方向上“无界”，对应到min方向上即为-inf，max方向上即为+inf
    """

    def __new__(cls, var_inp, *args, dtype=None) -> 'Bounds':
        return np.asanyarray(np.r_[var_inp, args], dtype=dtype).view(cls)

    def __array_finalize__(self, _, **__):
        pass

    @staticmethod
    def unbounded(n_axes=0) -> 'Bounds':
        """获取无界边界

        Parameters
        ----------
        n_axes
            维度

        Returns
        -------
        ret
            无界形式下的n维边界（全为np.nan）
        """
        return np.full(n_axes * 2, np.nan).view(Bounds)

    @staticmethod
    def bounded(xmin=np.nan, xmax=np.nan, ymin=np.nan, ymax=np.nan, zmin=np.nan, zmax=np.nan, *, n_axes=None):
        if n_axes is None:
            n_axes = np.where(~np.isnan([0, 0, xmin, xmax, ymin, ymax, zmin, zmax]))[0][-1] // 2

        ret = Bounds.unbounded(n_axes)

        if n_axes > 0:
            ret.xmin = xmin
            ret.xmax = xmax
        if n_axes > 1:
            ret.ymin = ymin
            ret.ymax = ymax
        if n_axes > 2:
            ret.zmin = zmin
            ret.zmax = zmax

        return ret

    partial = staticmethod(bounded)

    def as_corners(self):
        """从边界形式转换为角落点形式

        Returns
        -------
        ret
            角落点形式表示的
        """
        return self.reshape(-1, 2).T.view(Corners)

    def to_inf_format(self):
        """将无界条件从标准形式转换为正负无穷形式

        Returns
        -------
        ret
            正负无穷形式表示的边界（非标准形式）
        """
        ret = self.copy()
        infs = np.inf * np.tile([-1, 1], self.n_axes)
        mask = np.isnan(ret)
        ret[mask] = infs[mask]

        return ret

    def expand(self, *, proportion=None, increment=None, inplace=False):
        """扩展或收缩边界

        Parameters
        ----------
        proportion
            按对应轴长度的比例扩展边界，可以是一个数字，也可以是一个数组，支持通过 `Bounds.unbounded` 指定
        increment
            按固定增量扩展边界，可以是一个数字，也可以是一个数组，支持通过 `Bounds.unbounded` 指定
        inplace
            指示操作是否修改self

        Returns
        -------
        ret
            扩张后的边界对象

        Notes
        -----
        在-x, -y, -z方向上负数为增长，在+x, +y, +z方向上正数为增长

        如果某个轴向的 `扩展比例` 和 `扩展增量` 同时指定，则 `增量` 覆盖 `比例` 的对应数值

        >>> Bounds(1, 2, 3, 4).expand(proportion=0.5)
        <<< Bounds(0.5, 2.5, 2.5, 4.5)

        >>> Bounds(1, 2, 3, 4).expand(increment=1)
        <<< Bounds(0, 3, 2, 5)
        """
        assert proportion is not None or increment is not None, (
            'Either `proportion` or `increment` must be provided.'
        )

        deltas = np.tile(np.asarray((-1, 1)), self.n_axes)
        inc = Bounds.unbounded(self.n_axes)

        if proportion is not None:
            if np.ndim(proportion) == 0:
                proportion = deltas * proportion

            proportion = Bounds(proportion)
            n_axes = min(proportion.n_axes, self.n_axes)
            proportion_inc = self.extent[:n_axes].repeat(2) * proportion[:2 * n_axes]

            inc.override(by=proportion_inc, inplace=True)

        if increment is not None:
            if np.ndim(increment) == 0:
                increment = increment * deltas
            inc.override(by=increment, inplace=True)

        return self.override(by=self + inc, inplace=inplace)

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
        other = Bounds(by)

        dtype = np.common_type(self, other)
        if inplace:
            if dtype != self.dtype:
                raise TypeError(
                    f'Failed to change values ({self.dtype}) into incompatible type ({dtype}).'
                    f' Try disabling `inplace` to allow making copies.'
                )
            target = self
        else:
            target = self.astype(dtype, copy=True)

        length = min(target.n_axes, other.n_axes) * 2
        mask = ~np.isnan(other)[:length]
        target[:length][mask] = other[:length][mask]

        return target

    def set(self, axis, min=None, max=None):
        if np.size(min) == 2 and max is None:
            # 适配 b.set(0, [0, 10]) 用法
            min, max = min

        if min is not None:
            if np.isneginf(min):
                min = np.nan
            elif np.isposinf(min):
                raise ValueError(f'Min bound cannot be set to `+inf` in axis {axis}.')
            self[2 * axis] = min

        if max is not None:
            if np.isposinf(max):
                max = np.nan
            elif np.isneginf(max):
                raise ValueError(f'Max bound cannot be set to `-inf` in axis {axis}.')
            self[2 * axis + 1] = max

    def get(self, axis):
        return self[2 * axis: 2 * axis + 2]

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

    @property
    def center(self):
        return (self.origin + self.end) / 2


class Corners(FixedShapeNDArray):
    def __new__(cls, input_arr):
        return np.asanyarray(input_arr).view(cls)

    def as_bounds(self) -> Bounds:
        """从角落点形式转换为边界形式

        Returns
        -------
        ret
            角边界形式表示的
        """
        return self.T.ravel().view(Bounds)

    def to_inf_format(self) -> np.ndarray:
        """将无界条件从标准形式转换为正负无穷形式

        Returns
        -------
        ret
            正负无穷形式表示的角落点（非标准形式）
        """
        ret = np.array(self)
        infs = np.inf * np.asarray([-1, 1]).repeat(self.n_axes).reshape((2, -1))
        mask = np.isnan(ret)
        ret[mask] = infs[mask]

        return ret

    def set(self, axis, min=None, max=None):
        if np.size(min) == 2 and max is None:
            # 适配 c.set(0, [0, 10]) 用法
            min, max = min

        if min is not None:
            if np.isneginf(min):
                min = np.nan
            elif np.isposinf(min):
                raise ValueError(f'Min bound cannot be set to `+inf` in axis {axis}.')
            self[0, axis] = min

        if max is not None:
            if np.isposinf(max):
                max = np.nan
            elif np.isneginf(max):
                raise ValueError(f'Max bound cannot be set to `-inf` in axis {axis}.')
            self[1, axis] = max

    def get(self, axis):
        return self[:, axis]

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

    @property
    def center(self):
        return (self.origin + self.end) / 2


bounded = Bounds.bounded
unbounded = Bounds.unbounded
