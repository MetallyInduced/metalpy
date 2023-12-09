import discretize
import numpy as np
import pyvista as pv
from SimPEG import utils
from SimPEG.potential_fields import magnetics
from SimPEG.potential_fields.magnetics import analytics
from SimPEG.potential_fields.magnetics.simulation import Simulation3DIntegral, Simulation3DDifferential

from metalpy.carto.utils.mpl import plot_compare
from metalpy.scab import Tied, Demaged, Formatted, Fixed
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

    # 网格配置改自 SimPEG 测试用例 tests/pf/test_sensitivity_PFproblem.py
    # https://github.com/simpeg/simpeg/blob/main/tests/pf/test_sensitivity_PFproblem.py
    hxind = [(cs, 5, -1.3), (cs / 2.0, 41), (cs, 5, 1.3)]
    hyind = [(cs, 5, -1.3), (cs / 2.0, 41), (cs, 5, 1.3)]
    hzind = [(cs, 5, -1.3), (cs / 2.0, 40), (cs, 5, 1.3)]
    mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

    scene = Ellipsoid.sphere(sphere_rad).translated(*sphere_center).to_scene(model=chiblk)
    model_mesh = scene.build_model(mesh)

    # 通过 或运算符 来将观测点纳入网格范围，再通过 expand 来扩展网格边界
    # from metalpy.carto.coords import Coordinates
    # model_mesh = scene.build(
    #     cell_size=12,
    #     bounds=(scene.bounds | Coordinates(rxLoc).bounds).expand(proportion=[-0.5, 0.5, -0.5, 0.5, -1, 1])
    # )

    p = pv.Plotter()
    p.add_mesh(model_mesh.to_polydata(prune=False), opacity=0.5, show_edges=True)
    p.add_mesh(model_mesh.to_polydata())
    p.add_mesh(rxLoc)
    p.show()

    components = ["bx", "by", "bz", 'tmi']

    # integral simulation with demagnetization
    builder = SimulationBuilder.of(Simulation3DIntegral)
    builder.patched(Tied(), Demaged(), Formatted())
    builder.store_sensitivities(False)
    builder.source_field(Btot, Inc, Dec)
    builder.receivers(rxLoc, components)
    builder.chi_map()
    builder.active_mesh(model_mesh)
    sim_int = builder.build()

    # differential simulation
    builder = SimulationBuilder.of(Simulation3DDifferential)
    builder.patched(Formatted(), Fixed())
    builder.source_field(Btot, Inc, Dec)
    builder.receivers(rxLoc, components)
    builder.chi_map()
    builder.active_mesh(model_mesh)
    sim_diff = builder.build()

    print(f'Time consumption:')

    with timed('Simulation3DIntegral time: '):
        dpred_int = sim_int.dpred(model_mesh.model)

    with timed('Simulation3DDifferential time: '):
        dpred_diff = sim_diff.dpred(model_mesh.model)

    # analytical solution
    bxa, bya, bza = magnetics.analytics.MagSphereAnaFunA(
        *rxLoc.T, sphere_rad, *sphere_center, chiblk, b0, "secondary"
    )
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
    ratio_diff_int = abs((dpred_diff - dpred_int) / (dpred_int + 1e-5))
    pr = [50, 80, 95]
    pr_str = format_ratios(np.asarray(pr) / 100, digits=0)
    print(f'Simulation3DIntegral error ({pr_str}): {format_ratios(np.percentile(ratio_int, pr))}')
    print(f'Simulation3DDifferential error ({pr_str}): {format_ratios(np.percentile(ratio_diff, pr))}')
    print(f'Diff. / Int. difference ({pr_str}): {format_ratios(np.percentile(ratio_diff_int, pr))}')

    plot_compare(
        obs=rxLoc,
        data_arrays=[
            dpred_int[:, -1],
            dpred_diff[:, -1],
            dpred_int[:, -1] - dpred_diff[:, -1],
            dpred_ana[:, -1],
        ],
        plot_titles=['Integral', 'Differential', 'Absolute Difference', 'Analytical'],
        colorbar_titles='nT'
    )


def format_ratios(pr, digits=2):
    return ", ".join([f"{p:.{digits}%}" for p in pr])


if __name__ == '__main__':
    main()
