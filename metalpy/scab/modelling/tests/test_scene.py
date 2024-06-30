import numpy as np
from numpy.testing import assert_equal

from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Cuboid, Prism, Ellipsoid, Tunnel


def test_build():
    scene = Scene.of(
        Cuboid([1, 1, 1], size=2),
        Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3),
        Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3).rotated(90, 0, 0),
        Ellipsoid.spheroid(1, 3, 0).translated(0, -2, 2),
        Tunnel([-3, 2, 2], 0.5, 1, 3),
        models=[1, 2, 3, 4, 5],
    )

    bounds = np.round(scene.bounds)
    ret = [
        (scene.build(cell_size=0.2, bounds=bounds), 'cell sizes on all axes (reference)'),
        (scene.build(cell_size=[0.2, 0.2, 0.2], bounds=bounds), 'cell sizes on each axis'),
        (scene.build(n_cells=13500, bounds=bounds), 'n total cells'),
        (scene.build(n_cells=[30, 30, 15], bounds=bounds), 'n cells on each axis')
    ]
    ref_mesh = ret[0][0]

    for model_mesh, mesh_type in ret[1:]:
        assert_equal(
            model_mesh.n_cells, ref_mesh.n_cells,
            f'Mismatched total mesh cells (mesh built with {mesh_type})'
        )
        assert_equal(
            model_mesh.mesh.cell_centers, ref_mesh.mesh.cell_centers,
            f'Mismatched mesh cell centers (mesh built with {mesh_type})'
        )
        assert_equal(
            model_mesh.get_active_model(), model_mesh.get_active_model(),
            f'Mismatched mesh model (mesh built with {mesh_type})'
        )
