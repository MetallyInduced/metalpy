import numpy as np
import pytest

from metalpy.scab.modelling.shapes import Cuboid
from metalpy.scab.modelling.shapes.testing.shape3d_place_test import assert_placing_inside


def test_cuboid():
    assert_placing_inside(Cuboid(bounds=[1, 2, 3, 4, 5, 6]))


def test_cuboid_ctor():
    # 测试各种定义方式的等价性
    cubes = [
        Cuboid([0, 2, 0], [2, 0, 2]),
        Cuboid([0, 0, 0], [2, 2, 2]),
        Cuboid(origin=[0, 0, 0], size=[2, 2, 2]),
        Cuboid(origin=[0, 0, 0], size=2),
        Cuboid(origin=[0, 0, 0], center=[1, 1, 1]),
        Cuboid(end=[2, 2, 2], size=2),
        Cuboid(end=[2, 2, 2], center=[1, 1, 1]),
        Cuboid(center=[1, 1, 1], size=[2, 2, 2]),
        Cuboid(bounds=[0, 2, 0, 2, 0, 2]),
    ]

    origins = np.vstack([cube.origin for cube in cubes])
    lengths = np.vstack([cube.lengths for cube in cubes])

    assert np.all(origins == origins[0])
    assert np.all(lengths == lengths[0])


def test_cuboid_failure():
    # 测试严格模式下错误定义报错
    with pytest.raises(ValueError):
        _ = Cuboid([0, 2, 0], [2, 0, 2], no_corner_adjust=True)
