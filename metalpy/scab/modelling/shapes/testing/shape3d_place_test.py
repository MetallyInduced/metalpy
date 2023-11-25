import numpy as np

from metalpy.scab.modelling.shapes import Shape3D


def assert_placing_inside(shape: Shape3D, cell_size=1):
    """验证 `shape.place` 计算的所有有效网格均在模型体内。

    通过PyVista的体素化方法或隐式距离方法进行验证。
    """
    model_mesh = shape.build(cell_size=cell_size)

    try:
        active = Shape3D.do_place(shape, model_mesh.active_cell_centers, None)
    except (RuntimeError, AttributeError):
        # shape为MultiBlock或者其它异常情况导致默认的 `do_place` 无效
        active = shape.compute_implicit_distance(model_mesh.active_cell_centers) <= 0

    assert np.all(active)
