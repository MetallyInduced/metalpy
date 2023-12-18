import numpy as np
import pytest
from SimPEG.potential_fields import magnetics
from SimPEG.potential_fields.magnetics import analytics
from SimPEG.potential_fields.magnetics.simulation import Simulation3DIntegral, Simulation3DDifferential
from SimPEG.utils.mat_utils import dip_azimuth2cartesian
from discretize.utils import mkvc
from geoana.em.static import MagneticDipoleWholeSpace

from metalpy.carto.coords import Coordinates
from metalpy.mexin.utils.type_map import assert_all_types_exists
from metalpy.scab import Tied, Demaged, Formatted, Fixed
from metalpy.scab.builder import SimulationBuilder
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.scab.potential_fields.magnetics import Simulation3DDipoles
from metalpy.utils.sensor_array import get_grids_ex


@pytest.fixture
def rx_loc():
    return get_grids_ex(
        origin=[-200, -200, 150],
        end=[200, 200, 150],
        n=[5, 5, 1]
    ).pts


@pytest.fixture
def components():
    return ["bx", "by", "bz"]


def test_builders_map():
    assert_all_types_exists(SimulationBuilder._registry)


def test_builders(rx_loc, components):
    Inc, Dec, Btot = 45.0, 45.0, 51000
    b0 = analytics.IDTtoxyz(-Inc, Dec, Btot)

    sphere_rad = 100
    sphere_center = [0, 0, 0]
    chiblk = 3

    # 网格配置改自 SimPEG 测试用例 tests/pf/test_sensitivity_PFproblem.py
    # https://github.com/simpeg/simpeg/blob/main/tests/pf/test_sensitivity_PFproblem.py
    scene = Ellipsoid.sphere(sphere_rad).translated(*sphere_center).to_scene(model=chiblk)
    rx_bounds = Coordinates(rx_loc).bounds
    sim_bounds = (scene.bounds | rx_bounds).expand(proportion=10)

    model_mesh = scene.build(cell_size=40)
    model_mesh = model_mesh.expand(sim_bounds, ratio=1.3)  # 采用指数扩大网格对网格边界进行扩展

    # integral simulation with demagnetization
    builder = SimulationBuilder.of(Simulation3DIntegral)
    builder.patched(Tied(), Demaged(kernel_dtype=np.float32), Formatted())
    builder.source_field(Btot, Inc, Dec)
    builder.receivers(rx_loc, components)
    builder.chi_map()
    builder.active_mesh(model_mesh)
    builder.sensitivity_dtype(np.float32)
    builder.store_sensitivities(False)
    dpred_int = builder.build().dpred(model_mesh.model)

    # differential simulation
    builder = SimulationBuilder.of(Simulation3DDifferential)
    builder.patched(Formatted(), Fixed())
    builder.source_field(Btot, Inc, Dec)
    builder.receivers(rx_loc, components)
    builder.chi_map()
    builder.active_mesh(model_mesh)
    dpred_diff = builder.build().dpred(model_mesh.model)

    # analytical solution
    bxa, bya, bza = magnetics.analytics.MagSphereAnaFunA(
        *rx_loc.T, sphere_rad, *sphere_center, chiblk, b0, "secondary"
    )
    dpred_ana = np.c_[bxa, bya, bza]

    ratio_int = abs((dpred_int - dpred_ana) / (dpred_ana + 1e-5))
    ratio_diff = abs((dpred_diff - dpred_ana) / (dpred_ana + 1e-5))

    assert np.percentile(ratio_int, 80) < 0.3
    assert np.percentile(ratio_diff, 80) < 0.3


def test_dipoles_builder(rx_loc, components):
    dipole_location = [[0, 0, 0], [10, 10, 10]]
    moment = np.asarray([100, 3000])
    orientation = dip_azimuth2cartesian(*np.asarray([[10, 10], [45, -67]]).T)

    model = mkvc(moment[:, None] * orientation)

    # dipoles simulation
    builder = SimulationBuilder.of(Simulation3DDipoles)
    builder.patched(Formatted())
    builder.sources(dipole_location)
    builder.receivers(rx_loc, components)
    builder.store_sensitivities(False)
    dpred_dipoles = builder.build().dpred(model)

    # analytical solution
    dipoles = [
        MagneticDipoleWholeSpace(
            location=loc,
            orientation=ori,
            moment=mmt,
        )
        for loc, ori, mmt in zip(dipole_location, orientation, moment)
    ]
    dpred_ana = np.sum([dipole.magnetic_flux_density(rx_loc) for dipole in dipoles], axis=0)

    np.testing.assert_allclose(dpred_dipoles, dpred_ana)
