import numpy as np
from numpy.testing import assert_almost_equal

from metalpy.scab.modelling.transform import Rotation


def test_rotation():
    mesh = np.asarray([
        [0, 0, 1],
        [0, 0, 2],
        [0, 1, 0],
    ])
    rot = Rotation(30, 40, 50)
    transformed = rot.inverse_transform(rot.transform(mesh))
    assert_almost_equal(transformed, mesh)

    mesh = np.asarray([1, 0, 0])

    # 新版本Rotation下 a、b、y固定为x、y、z轴的旋转，seq只决定旋转坐标轴的先后顺序，不再改变a、b、y的含义
    rot = Rotation(-45, 45, 90, seq='yzx')
    assert_almost_equal(rot.transform(mesh), np.r_[0, 0, -1])
    assert_almost_equal(rot.inverse_transform(mesh), np.r_[0, -1, 0])
