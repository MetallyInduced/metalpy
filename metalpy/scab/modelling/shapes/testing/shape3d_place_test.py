import numpy as np

from metalpy.scab.modelling.shapes import Shape3D


def assert_placing_inside(shape: Shape3D, cell_size=1):
    model_mesh = shape.build(cell_size=cell_size)

    try:
        active = Shape3D.do_place(shape, model_mesh.active_cell_centers, None)
    except RuntimeError:
        active = shape.compute_implicit_distance(model_mesh.active_cell_centers) < 0

    assert np.all(active)
