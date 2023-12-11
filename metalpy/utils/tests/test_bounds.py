from numpy.testing import assert_almost_equal

from metalpy.utils.bounds import Bounds


def test_expand_by_proportion():
    base = Bounds(1, 2, 3, 5, 6, 9)

    # 通过单独数值指定
    assert_almost_equal(base.expand(proportion=0.5), [0.5, 2.5, 2, 6, 4.5, 10.5])

    # 通过数组指定
    assert_almost_equal(
        base.expand(proportion=[-0.1, 0.2, -0.3, 0.4, -0.5, 0.6]),
        [0.9, 2.2, 2.4, 5.8, 4.5, 10.8]
    )

    # 不完全指定
    assert_almost_equal(base.expand(proportion=[0.1, 0.2]), [1.1, 2.2, 3, 5, 6, 9])
    assert_almost_equal(base.expand(proportion=Bounds.bounded(zmin=0.5)), [1, 2, 3, 5, 7.5, 9])
    assert_almost_equal(base.expand(proportion=Bounds.unbounded()), base)


def test_expand_by_increment():
    base = Bounds(1, 2, 3, 5, 6, 9)

    # 通过单独数值指定
    assert_almost_equal(base.expand(increment=1), [0, 3, 2, 6, 5, 10])

    # 通过数组指定
    assert_almost_equal(base.expand(increment=[-1, 2, -3, 4, -5, 6]), [0, 4, 0, 9, 1, 15])

    # 不完全指定
    assert_almost_equal(base.expand(increment=[1, 2]), [2, 4, 3, 5, 6, 9])
    assert_almost_equal(base.expand(increment=Bounds.bounded(ymin=0.5)), [1, 2, 3.5, 5, 6, 9])
    assert_almost_equal(base.expand(increment=Bounds.unbounded()), base)

    # increment与proportion同时指定，increment覆盖proportion
    assert_almost_equal(base.expand(
        proportion=1,
        increment=Bounds.bounded(ymin=-100)
    ), [0, 3, -97, 7, 3, 12])
