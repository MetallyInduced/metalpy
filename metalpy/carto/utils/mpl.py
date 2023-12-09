import numpy as np
from SimPEG.utils import plot2Ddata

import matplotlib as mpl
from matplotlib import pyplot as plt, gridspec


def plot_compare(
    obs,
    data_arrays,
    plot_titles=(),
    colorbar_titles=(),
    *,
    fig=None,
    show=None
):
    n_plots = len(data_arrays)

    if np.ndim(colorbar_titles) == 0:
        colorbar_titles = [colorbar_titles] * n_plots

    if show is None:
        show = fig is None  # 如果没有指定fig，则默认show

    if fig is None:
        fig = plt.figure(figsize=(6 * n_plots, 4))

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

    plot_width = 10
    spacer = 4
    bar_width = 1
    gs = gridspec.GridSpec(plot_width, (plot_width + bar_width + spacer) * n_plots + 1)

    for ii in range(n_plots):
        gs_index = (plot_width + bar_width + spacer) * ii

        ax1[ii] = fig.add_subplot(
            gs[0:plot_width, gs_index:gs_index + plot_width]
        )
        cplot[ii] = plot2Ddata(
            obs,
            data_arrays[ii],
            ax=ax1[ii],
            ncontour=30,
            clim=(-v_lim[ii], v_lim[ii]),
            contourOpts={"cmap": "bwr"},
        )

        if ii < len(plot_titles):
            ax1[ii].set_title(plot_titles[ii])

        ax1[ii].set_xlabel("x (m)")
        ax1[ii].set_ylabel("y (m)")

        ax2[ii] = fig.add_subplot(
            gs[0:plot_width, gs_index + plot_width:gs_index + plot_width + bar_width]
        )
        norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
        cbar[ii] = mpl.colorbar.ColorbarBase(
            ax2[ii], norm=norm[ii], orientation="vertical", cmap=mpl.cm.bwr
        )

        if ii < len(colorbar_titles):
            # cbar[ii].set_label(plot_units[ii], loc='top', size=12)
            ax2[ii].set_title(colorbar_titles[ii])

    if show:
        fig.show()
