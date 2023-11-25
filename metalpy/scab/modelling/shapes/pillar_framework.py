from . import Shape3D, PlateFramework, Composition, BarFramework


class PillarFramework(PlateFramework):
    def __init__(self,
                 outline: Shape3D,
                 spec,
                 n_rooms=None,
                 room_spec=None,
                 bounds=None,
                 *,
                 inherit_transform=False
                 ):
        """定义一个楼板-框架结构，其边界由 `outline` 决定，并基于 `outline` 裁剪边界。
        """
        super().__init__(
            outline, spec, n_rooms, room_spec, bounds,
            inherit_transform=inherit_transform
        )

    def _create_base_framework(self):
        return Composition(
            *PlateFramework._create_parts(self, 2),
            *BarFramework._create_parts(self, 2),
            mix_mode=Composition.Union
        )
