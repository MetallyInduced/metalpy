import numpy as np

from metalpy.utils.polygon.ear_clip import is_inside_convex_polygon


def test_inside_polygon_tester():
    p = np.asarray([3, 3])

    def _test_polygon(shp, outside=False):
        assert is_inside_convex_polygon(p, shp) ^ outside
        assert is_inside_convex_polygon(p, shp[::-1]) ^ outside

    # 朴素内点
    _test_polygon([[0, 0], [6, 0], [3, 6]])

    # 朴素外点
    _test_polygon([[0, 0], [6, 0], [3, -6]], outside=True)

    # 朴素边界点
    _test_polygon([[0, 0], [6, 6], [6, 0]])

    # 朴素角点
    _test_polygon([[0, 0], [3, 3], [6, 0]])

    # 共线外点
    _test_polygon([[0, 0], [2, 2], [6, 0]], outside=True)

    # 全共线内点
    _test_polygon([[0, 0], [2, 2], [3, 3]])

    # 全共线外点
    _test_polygon([[0, 0], [2, 2], [1, 1]], outside=True)

    # 共点内点
    _test_polygon([[3, 3], [3, 3], [3, 3]])

    # 共点外点
    _test_polygon([[0, 0], [0, 0], [0, 0]], outside=True)
