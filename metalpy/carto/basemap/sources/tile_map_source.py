from abc import ABC, abstractmethod
from typing import Iterable

from metalpy.utils.arg_specs import ArgSpecs
from .tile_locator import TileLocator


class TileMapSource(ABC):
    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        arg_spec = ArgSpecs.of(cls)
        arg_spec.push_args(*args)
        arg_spec.bind_kwargs(**kwargs)
        repr_str = arg_spec.get_func_repr(cls)

        obj = super().__new__(cls)
        obj.repr = repr_str

        return obj

    @property
    def repr(self):
        return getattr(self, '__repr')

    @repr.setter
    def repr(self, val):
        setattr(self, '__repr', val)

    @abstractmethod
    def __iter__(self) -> Iterable[TileLocator]:
        """返回若干个tile资源定位器

        Returns
        -------
        iterable_of_tiles
            TileLocator的迭代器
        """
        pass

    def warp_coordinates(self, coords):
        """输入wgs84坐标系下的坐标，如果该地图源对坐标系有特殊需求，需要在此进行转换
        例如高德地图等中国地图需要转换为GCJ02坐标系

        Returns
        -------
        ret
            地图源所使用的坐标系下的坐标
        """
        return coords

    def __repr__(self):
        return self.repr
