import numpy as np
from SimPEG.utils import plot2Ddata

import matplotlib as mpl
from matplotlib import pyplot as plt


def plot_compare(
    obs,
    data_arrays,
    plot_titles,
    plot_units,
    *,
    fig=None
):
    n_plots = len(data_arrays)

    if np.ndim(plot_units) == 0:
        plot_units = [plot_units] * n_plots

    if fig is None:
        fig = plt.figure(figsize=(4 * n_plots + 3, 4))

    ax1 = n_plots * [None]
    ax2 = n_plots * [None]
    norm = n_plots * [None]
    cbar = n_plots * [None]
    cplot = n_plots * [None]
    v_lim = [np.max(np.abs(a)) for a in data_arrays]

    max_lim = max(v_lim)
    for i in range(n_plots):
        # 尝试统一部分色表轴范围
        if (max_lim - v_lim[i]) / max_lim < 0.2:
            v_lim[i] = max_lim

    for ii in range(n_plots):
        ax1[ii] = fig.add_axes([0.33 * ii + 0.03, 0.11, 0.25, 0.84])
        cplot[ii] = plot2Ddata(
            obs,
            data_arrays[ii],
            ax=ax1[ii],
            ncontour=30,
            clim=(-v_lim[ii], v_lim[ii]),
            contourOpts={"cmap": "bwr"},
        )
        ax1[ii].set_title(plot_titles[ii])
        ax1[ii].set_xlabel("x (m)")
        ax1[ii].set_ylabel("y (m)")

        ax2[ii] = fig.add_axes([0.33 * ii + 0.27, 0.11, 0.01, 0.84])
        norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
        cbar[ii] = mpl.colorbar.ColorbarBase(
            ax2[ii], norm=norm[ii], orientation="vertical", cmap=mpl.cm.bwr
        )
        cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)
    fig.show()
