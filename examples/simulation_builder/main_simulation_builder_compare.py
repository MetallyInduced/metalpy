import discretize
import numpy as np
from SimPEG import utils
from SimPEG.potential_fields import magnetics
from SimPEG.potential_fields.magnetics import analytics
from SimPEG.potential_fields.magnetics.simulation import Simulation3DIntegral, Simulation3DDifferential

from metalpy.carto.utils.mpl import plot_compare
from metalpy.scab import Tied, Demaged
from metalpy.scab.builder.simulation_builder import SimulationBuilder
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.utils.time import timed


def main():
    Inc, Dec, Btot = 45.0, 45.0, 51000
    b0 = analytics.IDTtoxyz(-Inc, Dec, Btot)

    sphere_rad = 100
    sphere_center = [0, 0, 0]
    cs = 25.0
    chiblk = 3

    xr = np.linspace(-300, 300, 41)
    yr = np.linspace(-300, 300, 41)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones((xr.size, yr.size)) * 150
    rxLoc = np.c_[utils.mkvc(X), utils.mkvc(Y), utils.mkvc(Z)]

    hxind = [(cs, 5, -1.3), (cs / 2.0, 41), (cs, 5, 1.3)]
    hyind = [(cs, 5, -1.3), (cs / 2.0, 41), (cs, 5, 1.3)]
    hzind = [(cs, 5, -1.3), (cs / 2.0, 40), (cs, 5, 1.3)]
    mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

    scene = Ellipsoid.sphere(sphere_rad).translated(*sphere_center).to_scene(model=chiblk)
    model_mesh = scene.build_model(mesh)

    components = ["bx", "by", "bz", 'tmi']
    n_obs, n_comp = rxLoc.shape[0], len(components)

    # differential simulation
    builder = SimulationBuilder.of(Simulation3DDifferential)
    builder.source_field(Btot, Inc, Dec)
    builder.receivers(rxLoc, components)
    builder.chi_map()
    builder.active_mesh(model_mesh)
    sim = builder.build()

    # integral simulation with demagnetization
    builder = SimulationBuilder.of(Simulation3DIntegral)
    builder.patched(Tied(), Demaged())
    builder.store_sensitivities(False)
    builder.source_field(Btot, Inc, Dec)
    builder.receivers(rxLoc, components)
    builder.chi_map()
    builder.active_mesh(model_mesh)
    sim2 = builder.build()

    bxa, bya, bza = magnetics.analytics.MagSphereAnaFunA(
        *rxLoc.T, sphere_rad, *sphere_center, chiblk, b0, "secondary"
    )

    print(f'Time consumption:')

    with timed('Simulation3DIntegral time: '):
        dpred_int = sim2.dpred(model_mesh.model).reshape(n_obs, n_comp)

    with timed('Simulation3DDifferential time: '):
        u = sim.fields(model_mesh.model)
        dpred_diff = sim.projectFields(u).reshape(n_comp, n_obs).T  # 输出和积分方程法不一致

    dpred_ana = np.c_[bxa, bya, bza]
    dpred_ana = np.c_[dpred_ana, np.sum(dpred_ana * b0 / Btot, axis=1)]

    print(f'Overall error:')

    err_int = np.linalg.norm(dpred_int - dpred_ana) / np.linalg.norm(dpred_ana)
    err_diff = np.linalg.norm(dpred_diff - dpred_ana) / np.linalg.norm(dpred_ana)
    print(f'Simulation3DIntegral error: {err_int:.2%}')
    print(f'Simulation3DDifferential error: {err_diff:.2%}')

    print(f'Average error:')

    ratio_int = abs((dpred_int - dpred_ana) / (dpred_ana + 1e-5))
    ratio_diff = abs((dpred_diff - dpred_ana) / (dpred_ana + 1e-5))
    pr = [50, 80, 95]
    pr_str = format_ratios(np.asarray(pr) / 100, digits=0)
    print(f'Simulation3DIntegral error ({pr_str}): {format_ratios(np.percentile(ratio_int, pr))}')
    print(f'Simulation3DDifferential error ({pr_str}): {format_ratios(np.percentile(ratio_diff, pr))}')

    plot_compare(
        obs=rxLoc,
        data_arrays=[
            dpred_int[:, -1],
            dpred_diff[:, -1],
            dpred_int[:, -1] - dpred_diff[:, -1],
            dpred_ana[:, -1],
        ],
        plot_titles=['Integral', 'Differential', 'Absolute Difference', 'Analytical'],
        plot_units='nT'
    )


def format_ratios(pr, digits=2):
    return ", ".join([f"{p:.{digits}%}" for p in pr])


if __name__ == '__main__':
    main()
