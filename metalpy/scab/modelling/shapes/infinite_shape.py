import warnings
from abc import ABC

from metalpy.utils.bounds import Bounds
from metalpy.utils.object_path import get_type_name
from . import Shape3D


class InfiniteShape(Shape3D, ABC):
    """用于描述一个无穷3D形状
    """

    @property
    def local_bounds(self):
        return Bounds.unbounded()

    @property
    def volume(self):
        warnings.warn(f'Trying to access `volume` of infinite shape `{get_type_name(self)}`.')
        return 0

    @property
    def area(self):
        warnings.warn(f'Trying to access `area` of infinite shape `{get_type_name(self)}`.')
        return 0

    def rotate(self, *args, **kwargs):
        warnings.warn(f'Trying to rotate infinite shape `{get_type_name(self)}`,'
                      f' which may lead to unexpected behavior related to bounds and other property.')
        return super().rotate(*args, **kwargs)

    def rotated(self, *args, **kwargs):
        warnings.warn(f'Trying to rotate infinite shape `{get_type_name(self)}`,'
                      f' which may lead to unexpected behavior related to bounds and other property.')
        return super().rotate(*args, **kwargs)

    def translate(self, *args, **kwargs):
        warnings.warn(f'Trying to translate infinite shape `{get_type_name(self)}`.')
        return super().translate(*args, **kwargs)

    def translated(self, *args, **kwargs):
        warnings.warn(f'Trying to translate infinite shape `{get_type_name(self)}`.')
        return super().translated(*args, **kwargs)
