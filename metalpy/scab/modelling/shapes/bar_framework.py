import warnings
from functools import cached_property
from itertools import product
from typing import Iterable

import numpy as np

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
    if np.ndim(spec) > 0:
        spec = list(spec)
        assert len(spec) == n_params
    else:
        spec = [spec] * n_params

    return np.asarray(spec)


class BarFramework(Composition):
    def __init__(self,
                 outline: Shape3D,
                 spec=None,
                 n_rooms=None,
                 room_spec=None,
                 bounds=None,
                 *,
                 inherit_transform=False,
                 bar_spec=None
                 ):
        """定义一个规则框架模型，其边界由 `outline` 决定，并基于 `outline` 裁剪边界。

        Parameters
        ----------
        outline
            边界模型，一般为Cuboid或者立方体Prism
        spec, bar_spec
            三个方向上条柱的粗细，传入单一值时代表同时定义三个方向
        n_rooms
            三个方向上的房间数，传入单一值时代表同时定义三个方向
        room_spec
            三个方向上的房间尺寸，传入单一值时代表同时定义三个方向。
            房间尺寸以条柱中心为基准计算，保证结果中的房间尺寸为最大的不大于room_spec的等距划分。
            例如x方向有效宽度为12m，约定房间尺寸为11m，则结果中x方向房间数为2，单个房间尺寸为6m。
        bounds
            覆盖边界模型的边界。
            如果 `inherit_transform=True` 则应提供 `outline` 本地坐标系下的边界坐标。
            否则应提供世界坐标系下的边界。
        inherit_transform
            指示框架模型是否从 `outline` 继承空间变换信息。
            为 `True` 时框架模型边界将随着 `outline` 进行变换（会因变换而不和坐标轴正交）。
            为 `False` 时框架模型边界不会随着 `outline` 进行变换（保持与坐标轴正交）。
        """
        if spec is not None:
            self.spec = spec
        elif bar_spec is not None:
            self.bar_spec = bar_spec
        else:
            raise AssertionError('`spec` must be specified.')

        if bounds is not None:
            bounds = Bounds.copy(bounds)
            if inherit_transform:
                # 通过 `Cuboid` 做坐标转换
                bounds = Cuboid(bounds=bounds).apply_transform(outline.transforms, inplace=True).bounds

            self.outline_bounds = bounds

        if n_rooms is None:
            # 假设使用房间的尺寸定义
            # 则保证生成的房间不大于指定的尺寸
            # ni = ceil(bi / ri)
            room_spec = clarify_spec(room_spec)
            # 约定条柱不会超出outline定义的边界
            # 因此从边界条柱的中心开始算，实际的有效空间大小有缩小
            sizes = outline.local_bounds.extent - self.spec
            self.n_rooms = np.ceil(sizes / room_spec).astype(int)
        else:
            self.n_rooms = np.copy(n_rooms)

        self.outline = outline
        self.inherit_transform = inherit_transform

        # Shape3D层面保证所有place的结果为正值，所以Min和Max可以分别作为交集和并集使用
        super().__init__(
            mix_mode=Composition.Intersects
        )

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, v):
        self._spec = clarify_spec(v, 3)

    @property
    def n_rooms(self):
        return self._n_rooms

    @n_rooms.setter
    def n_rooms(self, v):
        self._n_rooms = clarify_spec(v, 3)

    @property
    def n_bars(self):
        return self.n_rooms + 1

    @property
    def bar_spec(self):
        return self.spec

    @bar_spec.setter
    def bar_spec(self, v):
        warnings.warn('`bar_spec` has been deprecated, consider using `spec` intead.')
        self.spec = v

    @property
    def bar_radius(self):
        return self.bar_spec / 2

    @property
    def axis_span(self):
        return self.outline_bounds.view().reshape((-1, 2))

    def with_(self, spec=None, n_rooms=None, outer_bars_only=False, *, bar_spec=None):
        if spec is not None:
            self.spec = spec
        elif bar_spec is not None:
            self.bar_spec = bar_spec

        if n_rooms is not None:
            self.n_rooms = n_rooms
        elif outer_bars_only:
            self.n_rooms = 1

        return self

    def cast_to(self, framework_type):
        return self.extract(framework_type=framework_type)

    def __getitem__(self, indices):
        full = slice(None, None, None)

        n_rooms = np.copy(self.n_rooms)
        bounds = self.local_bounds

        for axis, slicer in enumerate(indices):
            if slicer == full:
                continue

            if np.isscalar(slicer):
                slicer = slice(slicer, slicer + 1)

            centers = self.get_centers(axis)
            r = self.bar_radius[axis]

            lower_bars = centers[:-1][slicer]
            upper_bars = centers[1:][slicer]
            bounds.set(
                axis,
                min=min(lower_bars) - r,
                max=max(upper_bars) + r,
            )

            n_rooms[axis] = len(lower_bars)

        return self.extract(n_rooms=n_rooms, bounds=bounds)

    def get_centers(self, axis):
        a0, a1 = self.axis_span[axis]
        ra = self.bar_radius[axis]
        na = self.n_bars[axis]
        return np.linspace(a0 + ra, a1 - ra, na)

    def extract(self, *, n_rooms=None, bounds=None, framework_type=None):
        if n_rooms is None:
            n_rooms = self.n_rooms

        if bounds is None:
            bounds = self.bounds

        if framework_type is None:
            framework_type = type(self)

        return framework_type(
            outline=self.outline,
            spec=self.spec,
            n_rooms=n_rooms,
            bounds=bounds
        )

    @cached_property
    def outline_bounds(self):
        """获取框架部分在局部坐标系下的边界，框架部分会拷贝轮廓outline的transform信息
        """
        if self.inherit_transform:
            return self.outline.local_bounds
        else:
            return self.outline.bounds

    @cached_property
    def bars(self):
        ret = self._create_base_framework()

        if self.inherit_transform:
            ret.transforms = self.outline.transforms.clone()

        return ret

    @cached_property
    def shapes(self):
        return [
            self.bars,
            self.outline
        ]

    def _create_parts(self, axis):
        # 长轴为c，坐标为a，b
        a, b, c = axes = np.r_[axis + 1, axis + 2, axis] % 3  # 三个轴的下标
        ra, rb = self.bar_radius[a], self.bar_radius[b]
        c0, c1 = self.axis_span[c]

        axes = axes[axes]

        for ax, bx in product(self.get_centers(a), self.get_centers(b)):
            origin = np.r_[ax - ra, bx - rb, c0][axes]
            end = np.r_[ax + ra, bx + rb, c1][axes]
            yield Cuboid(origin=origin, end=end)

    def _create_base_framework(self):
        return Composition(
            *self._create_parts(0),
            *self._create_parts(1),
            *self._create_parts(2),
            mix_mode=Composition.Union
        )
