import numpy as np

from . import Shape3D, Cuboid, BarFramework


class PlateFramework(BarFramework):
    def __init__(self,
                 outline: Shape3D,
                 spec,
                 n_rooms=None,
                 room_spec=None,
                 bounds=None,
                 *,
                 inherit_transform=False
                 ):
        """定义一个规则板结构，其边界由 `outline` 决定，并基于 `outline` 裁剪边界。

        See Also
        --------
        :class:`BarFramework` : 规则框架模型
        """
        super().__init__(
            outline, spec, n_rooms, room_spec, bounds,
            inherit_transform=inherit_transform
        )

    def _create_parts(self, axis):
        # axis为厚度轴，其余两个轴为平面轴
        a, b, c = axes = np.r_[axis, axis + 1, axis + 2] % 3  # 三个轴的下标
        ra = self.bar_radius[a]
        b0, b1 = self.axis_span[b]
        c0, c1 = self.axis_span[c]

        axes = axes[axes]

        for ax in self.get_centers(a):
            origin = np.r_[ax - ra, b0, c0][axes]
            end = np.r_[ax + ra, b1, c1][axes]
            yield Cuboid(origin=origin, end=end)
