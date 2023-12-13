import numpy as np
from SimPEG.potential_fields.magnetics import Simulation3DIntegral
from discretize.utils import mkvc
from numpy.testing import assert_almost_equal

from metalpy.scab.builder import SimulationBuilder
from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.scab.utils.misc import define_inducing_field
from metalpy.utils.sensor_array import get_grids_ex


magnetic_background_field = (50000, 78, 3)
chi = 2


def create_forward_settings():
    scene = Scene.of(
        Ellipsoid.spheroid(10, 30, 0),
    )
    model_mesh = scene.build(cell_size=1, cache=True)
    bounds = scene.bounds

    # observation
    obs = get_grids_ex(
        origin=[bounds.xmin, bounds.ymin, 50],
        end=[bounds.xmax, bounds.ymax, 50],
        n=[10 + 1, 10 + 1, 1]
    ).pts
    components = ['tmi']

    return model_mesh, obs, components


def create_simulation_builder(taichi=True):
    model_mesh, obs, components = create_forward_settings()

    builder = SimulationBuilder.of(Simulation3DIntegral)
    builder.receivers(obs, components)
    builder.active_mesh(model_mesh)
    builder.source_field(*magnetic_background_field)
    builder.model_type(scalar=True)
    builder.sensitivity_dtype(np.float64)

    if taichi:
        from metalpy.scab import Tied
        builder.patched(Tied(max_cpu_threads=1))

    return builder, model_mesh.model


def create_simpeg_simulation():
    from SimPEG import maps
    from SimPEG.potential_fields import magnetics

    source_field = define_inducing_field(*magnetic_background_field)
    model_mesh, receiver_points, components = create_forward_settings()

    nC = model_mesh.n_active_cells
    receiver_list = magnetics.receivers.Point(receiver_points, components=components)
    receiver_list = [receiver_list]

    inducing_field = magnetics.sources.SourceField(
        receiver_list=receiver_list, parameters=source_field
    )
    survey = magnetics.survey.Survey(inducing_field)
    model_map = maps.IdentityMap(nP=nC)
    simulation = magnetics.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=model_mesh.mesh,
        model_type="scalar",
        chiMap=model_map,
        ind_active=model_mesh.active_cells,
        store_sensitivities="forward_only",
    )

    return simulation, model_mesh.model


def test_taichi_patch():
    # Taichi and Builder
    builder, model = create_simulation_builder(taichi=True)
    pred_taichi = builder.build().dpred(model * chi)

    # SimPEG
    simulation, model = create_simpeg_simulation()
    pred_simpeg = simulation.dpred(model * chi)

    # SimPEG新旧算法偏差最大能到千分之一 （Tied插件使用的是旧版本正演算法）
    assert_almost_equal(pred_taichi / pred_simpeg, 1, decimal=4)


def test_vector_model():
    field = define_inducing_field(*magnetic_background_field)

    # scalar model
    builder, model = create_simulation_builder(taichi=True)
    builder.model_type(scalar=True)
    pred_scalar = builder.build().dpred(model * chi)

    # vector model
    builder, model = create_simulation_builder(taichi=True)
    builder.model_type(vector=True)
    pred_vector = builder.build().dpred(mkvc(
        model[:, np.newaxis] * field.unit_vector * chi
    ))

    assert_almost_equal(pred_vector / pred_scalar, 1, decimal=7)
