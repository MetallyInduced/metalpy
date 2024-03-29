import numpy as np
from SimPEG import maps
from SimPEG.potential_fields import magnetics
from discretize.utils import mkvc
from numpy.testing import assert_almost_equal

import metalpy
from metalpy.scab import simpeg_patched, Progressed
from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.scab.utils.misc import define_inducing_field
from metalpy.utils.time import timed


def main():
    scene = Scene.of(Ellipsoid.spheroid(2, 8, polar_axis=0))

    bounds = scene.bounds
    b0, b1 = bounds[::2], bounds[1::2]
    grid_size = 0.5
    model_mesh = scene.build(cell_size=grid_size)
    mesh = model_mesh.mesh
    ind_active = model_mesh.active_cells

    n_active = model_mesh.n_active_cells
    mag_susc = 1
    scalar_model = np.ones(n_active) * mag_susc
    vector_model = mkvc(np.c_[scalar_model, scalar_model * 0.5, scalar_model * 1.5])

    n_x_grids, n_y_grids = 101, 11
    obsx, obsy, obsz = np.meshgrid(np.linspace(b0[0], b1[0], n_x_grids), np.linspace(b0[1], b1[1], n_y_grids), 20)
    obsx, obsy, obsz = mkvc(obsx), mkvc(obsy), mkvc(obsz)
    obs = np.c_[obsx, obsy, obsz]
    components = ('tmi', 'bx', 'by', 'bz', 'bzz', 'bxx', 'bxy', 'bxz', 'byy', 'byz',)
    H = define_inducing_field(50000, 90, 0)

    def test_forward(store_sensitivities, model_type, *patches):
        with simpeg_patched(*patches,):
            if model_type == 'vector':
                model = vector_model
                model_map = maps.IdentityMap(nP=3 * n_active)
            else:
                model = scalar_model
                model_map = maps.IdentityMap(nP=n_active)

            receiver1 = magnetics.receivers.Point(obs, components=components[:4])
            receiver_list = [receiver1, *[magnetics.receivers.Point(obs, components=c) for c in components[4:]]]

            source_field = magnetics.sources.SourceField(
                receiver_list=receiver_list, parameters=H
            )
            survey = magnetics.survey.Survey(source_field)

            simulation = magnetics.simulation.Simulation3DIntegral(
                survey=survey,
                mesh=mesh,
                model_type=model_type,
                chiMap=model_map,
                ind_active=ind_active,
                store_sensitivities=store_sensitivities,
            )

        return simulation.dpred(model)

    taichi_patches = [metalpy.scab.Tied('cpu'), Progressed()]

    with timed('| Scalar model | With sensitivity matrix | Taichi | : '):
        dpred_ti = test_forward('ram', 'scalar', *taichi_patches)

    with timed('| Scalar model | With sensitivity matrix | vanilla | : '):
        dpred_raw = test_forward('ram', 'scalar')

    assert_almost_equal(dpred_ti, dpred_raw, decimal=3)

    with timed('| Scalar model | Forward only | Taichi | : '):
        dpred_ti = test_forward('forward_only', 'scalar', *taichi_patches)

    with timed('| Scalar model | Forward only | vanilla | : '):
        dpred_raw = test_forward('forward_only', 'scalar')

    assert_almost_equal(dpred_ti, dpred_raw, decimal=3)

    with timed('| Vector model | With sensitivity matrix | Taichi | : '):
        dpred_ti = test_forward('ram', 'vector', *taichi_patches)

    with timed('| Vector model | With sensitivity matrix | vanilla | : '):
        dpred_raw = test_forward('ram', 'vector')

    assert_almost_equal(dpred_ti, dpred_raw, decimal=3)

    with timed('| Vector model | Forward only | Taichi | : '):
        dpred_ti = test_forward('forward_only', 'vector', *taichi_patches)

    with timed('| Vector model | Forward only | vanilla | : '):
        dpred_raw = test_forward('forward_only', 'vector')

    assert_almost_equal(dpred_ti, dpred_raw, decimal=3)


if __name__ == '__main__':
    main()
