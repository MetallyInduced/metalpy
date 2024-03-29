import numpy as np

from metalpy.scab.modelling.shapes import Shape3D


def assert_placing_inside(shape: Shape3D, cell_size=1):
    """验证 `shape.place` 计算的所有有效网格均在模型体内。

    通过PyVista的体素化方法或隐式距离方法进行验证。
    """
    model_mesh = shape.build(cell_size=cell_size)  # 调用子类重写的 do_place

    try:
        active = shape._do_place_pyvista(model_mesh.active_cell_centers)  # 调用基于PyVista的默认 do_place 实现
        if np.count_nonzero(active) < len(active):
            # 默认实现可能会遗漏模型面上的点，尝试通过隐式距离进一步判断计算
            active[~active] = place_by_implicit_distance(shape, model_mesh.active_cell_centers[~active])
    except (RuntimeError, AttributeError):
        # shape为MultiBlock或者其它异常情况导致默认的 `_do_place` 无效
        active = place_by_implicit_distance(shape, model_mesh.active_cell_centers)

    assert np.all(active), (f'`{type(shape).__name__}.place` does not yield same result'
                            f' as default PyVista implementation.')


def place_by_implicit_distance(shape, cell_centers):
    return shape.compute_implicit_distance(cell_centers) <= 0
