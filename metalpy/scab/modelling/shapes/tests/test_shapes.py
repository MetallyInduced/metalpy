from metalpy.scab.modelling.shapes import Ellipsoid, Tunnel, Prism
from metalpy.scab.modelling.shapes.testing.shape3d_place_test import assert_placing_inside


def test_ellipsoid():
    assert_placing_inside(Ellipsoid(2, 3, 4))


def test_tunnel():
    assert_placing_inside(Tunnel([0, 0, 0], 3, 6, 3))


def test_prism():
    assert_placing_inside(Prism([
        [0, 0], [1, 1], [0, 2], [2, 2], [2, 0]
    ], 3, 4))
