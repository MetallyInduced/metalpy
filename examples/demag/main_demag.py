import matplotlib as mpl
import numpy as np
import taichi as ti
from SimPEG.potential_fields import magnetics
from SimPEG.utils import plot2Ddata
from discretize.utils import mkvc
from matplotlib import pyplot as plt

from metalpy.mepa import LinearExecutor
from metalpy.scab import Progressed, Tied
from metalpy.scab.builder import SimulationBuilder
from metalpy.scab.builder.potential_fields.magnetics import Simulation3DIntegralBuilder
from metalpy.scab.demag import Demagnetization
from metalpy.scab.demag.factored_demagnetization import FactoredDemagnetization
from metalpy.scab.demag.utils import get_prolate_spheroid_demag_factor
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.scab.tied.potential_fields.magnetics.simulation import TiedSimulation3DIntegralMixin
from metalpy.scab.utils.misc import define_inducing_field
from metalpy.utils.taichi import ti_prepare, ti_config
from metalpy.utils.time import Timer


def main(cell_size, gpu=False):
    if gpu:
        ti_prepare(arch=ti.gpu, device_memory_fraction=0.8)

    a, c = 10, 40
    timer = Timer()

    with timer:
        model_mesh = Ellipsoid.spheroid(a, c, polar_axis=0).to_scene(model=80).build(cell_size=cell_size)
        active_cells = model_mesh.active_cells
        model = model_mesh.get_active_model()
        mesh = model_mesh.mesh

        source_field = define_inducing_field(50000, 45, 20)

    print(f"Modelling: {timer}")

    with timer:
        # analytical demagnetization factor
        N = get_prolate_spheroid_demag_factor(c / a, polar_axis=0)
        demag = FactoredDemagnetization(n=N)
        demaged_model = demag.dpred(model, source_field=source_field)

        # numerical demagnetization factor
        with ti_config(arch=ti.gpu if gpu else ti.cpu):
            compression = {}
            if gpu:
                compression['method'] = Demagnetization.Compressed
                compression['compressed_size'] = 400000

            demag2 = Demagnetization(
                source_field=source_field,
                mesh=mesh,
                active_ind=active_cells,
                progress=True,
                **compression
            )
            demaged_model2 = demag2.dpred(model)

    print(f"Solving: {timer}")
    print("MagModel MAPE(%): ", (abs(demaged_model2 - demaged_model) / abs(demaged_model)).mean() * 100)

    obsx = np.linspace(-2048, 2048, 128 + 1)
    obsy = np.linspace(-2048, 2048, 128 + 1)
    obsx, obsy = np.meshgrid(obsx, obsy)
    obsx, obsy = mkvc(obsx), mkvc(obsy)
    obsz = np.full_like(obsy, 80)
    receiver_points = np.c_[obsx, obsy, obsz]

    # 手动引入来帮助Dask定位并上传到worker
    _ = TiedSimulation3DIntegralMixin, Simulation3DIntegralBuilder
    builder = SimulationBuilder.of(magnetics.simulation.Simulation3DIntegral)
    builder.patched(Tied(arch='gpu' if gpu else 'cpu'), Progressed())
    builder.source_field(*source_field)
    builder.receivers(receiver_points)
    builder.active_mesh(model_mesh)
    builder.model_type(vector=True)
    builder.store_sensitivities(False)
    simulation = builder.build()

    pred = simulation.dpred(mkvc(demaged_model))
    pred2 = simulation.dpred(mkvc(demaged_model2))

    return demaged_model, pred, demaged_model2, pred2, receiver_points


if __name__ == '__main__':
    executor = LinearExecutor(1)

    workers = [w for w in executor.get_workers() if 'large-mem' in w.group][:1]
    if len(workers) == 1:
        f = executor.submit(main, [1, 1, 0.5], workers=workers)
    else:
        ti_prepare(device_memory_fraction=0.9)
        gpu = False
        if executor.is_local():
            gpu = True
        f = executor.submit(main, [1.6, 1.6, 1], gpu=gpu)

    demaged_model, pred, demaged_model2, pred2, receiver_points = executor.gather(f)

    print('Model MAPE (%):', (abs(demaged_model2 - demaged_model) / abs(demaged_model)).mean() * 100)
    ape = abs((pred - pred2) / pred)
    print('TMI MAPE (%):', ape[abs(pred) > 1e-2].mean() * 100)

    fig = plt.figure(figsize=(17, 4))

    data_array = np.c_[pred, pred2, pred-pred2]
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
