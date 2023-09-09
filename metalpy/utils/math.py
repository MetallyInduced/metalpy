import numpy as np


def weighted_percentile(a, weights, p):
    from scipy.interpolate import interp1d

    weighted = [(x, w) for x, w in zip(a, weights)]
    weighted.sort(key=lambda xw: xw[0])
    xs, ws = np.asarray(weighted).T
    ws /= ws.sum() / 100
    ps = np.r_[0, ws.cumsum()]
    ps = ps[:-1] + np.diff(ps) / 2

    interp = interp1d(ps, xs, fill_value='extrapolate')

    return interp(p)
