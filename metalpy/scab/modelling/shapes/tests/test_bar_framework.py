from metalpy.scab.modelling.shapes import BarFramework, Ellipsoid
from metalpy.scab.modelling.shapes.testing.shape3d_place_test import assert_placing_inside


def shp():
    return Ellipsoid(10, 10, 10)


def _test_bar_framework(inherit_transform):
    kwargs = {
        'spec': 1,
        'n_rooms': 2,
        'inherit_transform': inherit_transform
    }
    assert_placing_inside(BarFramework(shp().rotated(0, 0, 45).translated(10, 10, 10), **kwargs))
    assert_placing_inside(BarFramework(shp(), **kwargs).rotated(0, 0, 45).translated(10, 10, 10))
    assert_placing_inside(BarFramework(shp().rotated(0, 0, 45), **kwargs).translate(10, 10, 10))


def test_bar_framework():
    _test_bar_framework(True)


def test_bar_framework_inherit_transform():
    _test_bar_framework(False)
