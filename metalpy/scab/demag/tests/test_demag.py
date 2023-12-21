import numpy as np
from SimPEG.potential_fields import magnetics
from discretize.utils import mkvc

from metalpy.scab import Tied, Demaged
from metalpy.scab.builder import SimulationBuilder
from metalpy.scab.demag.demagnetization import Demagnetization
from metalpy.scab.demag.utils import get_prolate_spheroid_demag_factor
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.utils.file import make_cache_directory
from metalpy.utils.numeric import limit_significand


def test_numerical_solvers():
    # 基于各种求解器求解退磁效应
    pred = {}
    for method in Demagnetization.methods:
        pred[method] = predict_with_demag(method=method)

    ref = Demagnetization.Compressed
    tmi_ref, model_ref = pred.pop(ref)
    for method, (tmi_pred, model_pred) in pred.items():
        # 测试不同方法结果是否有偏差
        a_and_b = f'`{ref}` and `{method}`'
        np.testing.assert_allclose(
            tmi_pred, tmi_ref, err_msg=f'TMI MAPE between {a_and_b}.',
            rtol=1e-2
        )
        np.testing.assert_allclose(
            model_pred, model_ref, err_msg=f'Demag model MAPE between {a_and_b}.',
            rtol=1e-1
        )


def test_numerical_solution():
    # 基于椭球体退磁因子计算
    tmi_truth, model_truth = predict_with_demag(by_factor=True)

    # 基于积分方程法求解退磁效应
    tmi_pred, model_pred = predict_with_demag(method=Demagnetization.Seperated)

    # 测试压缩求解器的结果是否和解析解有偏差
    tmi_mape = np.mean(np.abs((tmi_truth - tmi_pred) / tmi_truth))
    assert tmi_mape < 0.2

    model_mape = np.mean(np.abs((model_truth - model_pred) / model_truth))
    assert model_mape < 0.35


def predict_with_demag(
        by_factor=False,
        method=None,
        cell_size=(16, 4, 4)
):
    cell_size = limit_significand(cell_size)
    shape, factor = make_test_shape_and_demag_factor()
    model_mesh = shape.to_scene(model=80).build(cell_size=cell_size)

    if by_factor:
        demag = Demaged(factor=factor)
    else:
        demag = Demaged(method=method, compressed_size=0.0316)

    builder = SimulationBuilder.of(magnetics.simulation.Simulation3DIntegral)
    builder.patched(Tied(arch='cpu'), demag)
    builder.source_field(50000, 45, 20)
    builder.receivers(make_test_pts())
    builder.active_mesh(model_mesh)
    builder.model_type(scalar=True)
    builder.store_sensitivities(make_cache_directory(f'demag_sens/{cell_size}'))
    simulation = builder.build()
    tmi_pred = simulation.dpred(model_mesh.model)
    model_pred = simulation.chi

    return tmi_pred, model_pred


def make_test_shape_and_demag_factor():
    a, c = 10, 40
    shape = Ellipsoid.spheroid(a, c, polar_axis=0)
    factor = get_prolate_spheroid_demag_factor(c / a, polar_axis=0)

    return shape, factor


def make_test_pts():
    obsx = np.linspace(-512, 512, 64 + 1)
    obsy = np.linspace(-512, 512, 64 + 1)
    obsx, obsy = np.meshgrid(obsx, obsy)
    obsx, obsy = mkvc(obsx), mkvc(obsy)
    obsz = np.full_like(obsy, 80)
    receiver_points = np.c_[obsx, obsy, obsz]

    return receiver_points
