import contextlib


@contextlib.contextmanager
def check_axis(ax, *, show=True, figure=False, **fig_kw):
    """检查外部传入的axis是否有效，无效则基于 `fig_kw` 的参数调用 `plt.subplots` 创建fig和ax进行绘图

    Parameters
    ----------
    ax
        外部传入的matplotlib axis
    show
        指示如果是内部创建的figure，是否自动展示图像
    figure
        指示是否返回创建的figure，或者使用外部ax时返回None
    fig_kw
        用于创建新figure的参数，例如 `figsize`

    Notes
    -----
    如果 `figure=True`，则返回创建的figure和axis。

    如果 `figure=False`，则只返回axis。

    Examples
    --------
    >>> def plot_something(ax=None, show=True):
    >>>     with check_axis(ax, show=show, figsize=(5, 5)) as ax:
    >>>         ax.scatter([1, 2, 3, 4], [1, 6, 9, 16])

    >>> def plot_and_save(ax=None, fig=None):
    >>>     with check_axis(ax, show=False, figsize=(5, 5)) as fig, ax:
    >>>         ax.scatter([1, 2, 3, 4], [1, 6, 9, 16])
    >>>         if fig:
    >>>             fig.savefig('./screenshot.png')
    """
    from matplotlib import pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(**fig_kw)
    elif ax is plt:
        fig = plt

    if figure:
        yield fig, ax
    else:
        yield ax

    if show and fig is not None:
        fig.show()
