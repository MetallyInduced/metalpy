from functools import cached_property
from itertools import product
from typing import Iterable

import numpy as np

from metalpy.scab.modelling.mix_modes import MixMode
from metalpy.utils.bounds import Bounds
from . import Shape3D, Cuboid, Composition


def clarify_spec(spec, n_params=3):
    """定义一个长度为n_params的数组

    如果spec为可迭代类型，则验证spec长度是否满足。

    如果spec不为可迭代类型，则认为用单个spec定义了数组中的所有值，返回[spec] * n_params

    Parameters
    ----------
    spec
        待验证的参数
    n_params
        期望的参数个数

    Returns
    -------
    ret
        长度为n_params的数组，其值由spec定义
    """
    if isinstance(spec, Iterable):
        spec = list(spec)
        assert len(spec) == n_params
    else:
        spec = [spec] * n_params

    return np.asarray(spec)


class BarFramework(Composition):
    def __init__(self, outline: Shape3D, bar_spec, n_rooms=None, room_spec=None, bounds=None):
        """
        Parameters
        ----------
        outline
            边界，一般为Cuboid或者立方体Prism
        bar_spec
            三个方向上条柱的粗细，传入单一值时代表同时定义三个方向
        n_rooms
            三个方向上的房间数，传入单一值时代表同时定义三个方向
        room_spec
            三个方向上的房间尺寸，传入单一值时代表同时定义三个方向。
            房间尺寸以条柱中心为基准计算，保证结果中的房间尺寸为最大的不大于room_spec的等距划分。
            例如x方向有效宽度为12m，约定房间尺寸为11m，则结果中x方向房间数为2，单个房间尺寸为6m。
        """
        self.bar_spec = bar_spec = clarify_spec(bar_spec)

        if bounds is not None:
            self.framework_bounds = Bounds(bounds)

        if n_rooms is None:
            # 假设使用房间尺寸定义
            # 则保证生成的房间不大于指定的尺寸
            # ni = ceil(bi / ri)
            room_spec = clarify_spec(room_spec)
            # 约定条柱不会超出outline定义的边界
            # 因此从边界条柱的中心开始算，实际的有效空间大小有缩小
            sizes = outline.local_bounds.extent - bar_spec
            n_rooms = np.ceil(sizes / room_spec).astype(int)
        else:
            n_rooms = clarify_spec(n_rooms)

        self._n_rooms = n_rooms
        self.outline = outline

        # Shape3D层面保证所有place的结果为正值，所以Min和Max可以分别作为交集和并集使用
        super().__init__(
            mix_mode=Composition.Intersects
        )

    def with_(self, bar_spec=None, n_rooms=None, outer_bars_only=False):
        if bar_spec is not None:
            self.bar_spec = clarify_spec(bar_spec, 3)

        if n_rooms is not None:
            self._n_rooms = clarify_spec(n_rooms, 3)
        elif outer_bars_only:
            self._n_rooms = clarify_spec(1, 3)

        return self

    def __getitem__(self, indices):
        full = slice(None, None, None)

        n_rooms = self.n_rooms()
        bounds = self.local_bounds

        for axis, slicer in enumerate(indices):
            if slicer == full:
                continue

            if np.isscalar(slicer):
                slicer = slice(slicer, slicer + 1)

            centers = self.centers(axis)
            r = self.bar_radius(axis)

            lower_bars = centers[:-1][slicer]
            upper_bars = centers[1:][slicer]
            bounds.set(
                axis,
                min=min(lower_bars) - r,
                max=max(upper_bars) + r,
            )

            n_rooms[axis] = len(lower_bars)

        return BarFramework(
            outline=self.outline,
            bar_spec=self.bar_spec,
            n_rooms=n_rooms,
            bounds=bounds
        )

    def bar_radius(self, axis):
        return self.bar_spec[axis] / 2

    def n_rooms(self, axis=None):
        if axis is None:
            return self._n_rooms.copy()
        else:
            return self._n_rooms[axis]

    def n_bars(self, axis=None):
        return self.n_rooms(axis) + 1

    def axis_span(self, axis):
        return self.framework_local_bounds[axis * 2], self.framework_local_bounds[axis * 2 + 1]

    def centers(self, axis):
        a0, a1 = self.axis_span(axis)
        ra = self.bar_radius(axis)
        na = self.n_bars(axis)
        return np.linspace(a0 + ra, a1 - ra, na)

    def _create_bars(self, axis):
        # 长轴为c，坐标为a，b
        a, b, c = axes = np.r_[axis + 1, axis + 2, axis] % 3  # 三个轴的下标
        ra, rb = self.bar_radius(a), self.bar_radius(b)
        c0, c1 = self.axis_span(c)

        axes = axes[axes]

        for ax, bx in product(self.centers(a), self.centers(b)):
            origin = np.r_[ax - ra, bx - rb, c0][axes]
            end = np.r_[ax + ra, bx + rb, c1][axes]
            yield Cuboid(origin=origin, end=end)

    @cached_property
    def framework_local_bounds(self):
        """获取框架部分在局部坐标系下的边界，框架部分会拷贝轮廓outline的transform信息
        """
        return self.outline.local_bounds

    @cached_property
    def bars(self):
        ret = Composition(
            *self._create_bars(0),
            *self._create_bars(1),
            *self._create_bars(2),
            mix_mode=Composition.Union
        )
        ret.transforms = self.outline.transforms.clone()

        return ret

    @cached_property
    def shapes(self):
        return [
            self.bars,
            self.outline
        ]
