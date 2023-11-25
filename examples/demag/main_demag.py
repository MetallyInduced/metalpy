import copy

import matplotlib as mpl
import numpy as np
import taichi as ti
from SimPEG.potential_fields import magnetics
from SimPEG.utils import plot2Ddata
from discretize.utils import mkvc
from matplotlib import pyplot as plt

from metalpy.scab import Progressed, Tied, Demaged
from metalpy.scab.builder import SimulationBuilder
from metalpy.scab.demag.demagnetization import Demagnetization
from metalpy.scab.demag.utils import get_prolate_spheroid_demag_factor
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.scab.utils.misc import define_inducing_field
from metalpy.utils.numeric import limit_significand
from metalpy.utils.taichi import ti_prepare


def compute(cell_size, gpu=False):
    if gpu:
        ti_prepare(arch=ti.gpu, device_memory_fraction=0.8)

    cell_size = limit_significand(cell_size)
    a, c = 10, 40

    model_mesh = Ellipsoid.spheroid(a, c, polar_axis=0).to_scene(model=80).build(cell_size=cell_size)
    source_field = define_inducing_field(50000, 45, 20)

    obsx = np.linspace(-2048, 2048, 128 + 1)
    obsy = np.linspace(-2048, 2048, 128 + 1)
    obsx, obsy = np.meshgrid(obsx, obsy)
    obsx, obsy = mkvc(obsx), mkvc(obsy)
    obsz = np.full_like(obsy, 3 * a)
    receiver_points = np.c_[obsx, obsy, obsz]

    builder = SimulationBuilder.of(magnetics.simulation.Simulation3DIntegral)
    builder.patched(Tied(arch='gpu' if gpu else 'cpu'), Progressed())
    builder.source_field(*source_field)
    builder.receivers(receiver_points)
    builder.active_mesh(model_mesh)
    builder.model_type(scalar=True)
    builder.store_sensitivities(False)

    # 基于积分方程法求解退磁效应
    sim_numeric = copy.deepcopy(builder)
    sim_numeric.patched(Demaged(
        method=Demagnetization.Compressed,
        compressed_size=0.05,
        quantized=True,
        progress=True
    ))
    simulation = sim_numeric.build()
    pred_sim = simulation.dpred(model_mesh.model)
    demaged_model_sim = simulation.chi

    # 基于椭球体退磁因子计算
    sim_factored = copy.deepcopy(builder)
    factor = get_prolate_spheroid_demag_factor(c / a, polar_axis=0)
    sim_factored.patched(Demaged(factor=factor))
    simulation = sim_factored.build()
    pred_truth = simulation.dpred(model_mesh.model)
    demaged_model_truth = simulation.chi

    print(model_mesh.n_active_cells)

    return demaged_model_truth, pred_truth, demaged_model_sim, pred_sim, receiver_points


if __name__ == '__main__':
    # taichi相关配置
    ti_prepare(device_memory_fraction=0.9)
    gpu = False

    model_t, pred_t, model_p, pred_p, receiver_points = compute(
        cell_size=[2, 1, 1],
        gpu=gpu
    )

    model_mape = abs((model_p - model_t) / model_t).mean()
    tmi_mape = abs((pred_t - pred_p) / pred_t).mean()
    print(f'--- Mean Absolute Percentage Error ---')
    print(f'Model: {model_mape:.2%}')
    print(f'TMI: {tmi_mape:.2%}')

    fig = plt.figure(figsize=(17, 4))

    data_array = np.c_[pred_t, pred_p, pred_t - pred_p]
    plot_title = ["Observed", "Predicted", "Absolute Error"]
    plot_units = ["nT", "nT", ""]

    ax1 = 3 * [None]
    ax2 = 3 * [None]
    norm = 3 * [None]
    cbar = 3 * [None]
    cplot = 3 * [None]
    v_lim = [np.max(np.abs(data_array[:, :-1])),
             np.max(np.abs(data_array[:, :-1])),
             np.max(np.abs(data_array[:, -1]))]

    for ii in range(0, 3):
        ax1[ii] = fig.add_axes([0.33 * ii + 0.03, 0.11, 0.25, 0.84])
        cplot[ii] = plot2Ddata(
            receiver_points,
            data_array[:, ii],
            ax=ax1[ii],
            ncontour=30,
            clim=(-v_lim[ii], v_lim[ii]),
            contourOpts={"cmap": "bwr"},
        )
        ax1[ii].set_title(plot_title[ii])
        ax1[ii].set_xlabel("x (m)")
        ax1[ii].set_ylabel("y (m)")

        ax2[ii] = fig.add_axes([0.33 * ii + 0.27, 0.11, 0.01, 0.84])
        norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
        cbar[ii] = mpl.colorbar.ColorbarBase(
            ax2[ii], norm=norm[ii], orientation="vertical", cmap=mpl.cm.bwr
        )
        cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)
    plt.show()
