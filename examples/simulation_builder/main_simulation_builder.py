from SimPEG.potential_fields.magnetics.simulation import Simulation3DIntegral

from metalpy.scab import Tied, Progressed, Formatted
from metalpy.scab.builder.simulation_builder import SimulationBuilder
from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.utils.bounds import Bounds
from metalpy.utils.file import make_cache_directory
from metalpy.utils.sensor_array import get_grids_ex


def main():
    # scene
    scene = Scene.of(
        Ellipsoid.spheroid(10, 30, 0),
    )
    bounds = Bounds(-50, 50, -20, 20, -20, 20)
    model_mesh = scene.build(cell_size=1, bounds=bounds)
    scene.to_multiblock().plot(show_grid=True, color='white')
    model_mesh.to_polydata().threshold(0.5).plot(color='white', show_grid=True, show_edges=True)

    # observation
    obs = get_grids_ex(
        origin=[bounds.xmin, bounds.ymin, 50],
        end=[bounds.xmax, bounds.ymax, 50],
        n=[16 + 1, 16 + 1, 1]
    ).pts
    components = ['tmi']

    # simulation
    builder = SimulationBuilder.of(Simulation3DIntegral)
    builder.patched(Tied(), Progressed(), Formatted(pandas=True, locations=True))
    builder.receivers(obs, components)
    builder.scalar_model()
    builder.active_mesh(model_mesh)
    builder.store_sensitivities(make_cache_directory('sensitivities'))
    simulation = builder.build()

    # model and forward
    susceptibility = 3
    pred = simulation.dpred(model_mesh.model * susceptibility)

    print(pred)


if __name__ == '__main__':
    main()
