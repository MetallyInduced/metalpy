from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import tqdm
from scipy import stats

from metalpy.utils.file import file_cached
from metalpy.utils.math import weighted_percentile


def check_slope(x, y):
    """检查x和y的方差，变化更大的方向作为x，将计算的斜率定义域限定在[-pi/2, pi/2]

    Parameters
    ----------
    x, y
        点位置的x、y坐标

    Returns
    -------
    x, y
        如果x方向方差大于y，则返回x，y，否则搜索

    Notes
    -----
    该函数假定了点位置中绝大多数时间航向相同或相反
    """
    mid = len(x) // 2
    if np.std(x[mid - 50:mid + 50]) < np.std(y[mid - 50:mid + 50]):
        return y, x
    else:
        return x, y


def apply_delta_to_segments(segments, delta):
    return [(a + delta, b + delta) for a, b in segments]


def regulate_delta_directions(directions, inplace=False):
    if np.ndim(directions) == 0:
        if inplace:
            warnings.warn('Can not operate inplace-ly on a single element'
                          ' (with `inplace = True`).')
        if directions > 90:
            directions -= 180
        if directions < -90:
            directions += 180
        return directions

    if not inplace:
        directions = np.copy(directions)

    directions[directions > 90] -= 180
    directions[directions < -90] += 180  # 保证偏转角连续

    return directions


def diff_directions(directions):
    diff_direction = np.diff(directions)
    regulate_delta_directions(diff_direction)

    return diff_direction


@file_cached
def analyze_directions(x, y,
                       slices: Iterable[list | tuple | slice] | None = None,
                       robust=False,
                       progress=False):
    if slices is None:
        slices = [slice(None)]

    if progress:
        slices = tqdm.tqdm(slices)

    directions = []
    for s in slices:
        if isinstance(s, (tuple, list)):
            s = slice(*s)

        if np.unique(x[s], axis=0).shape[0] == 1:
            unique_ys = np.unique(y[s], axis=0)
            n_ys = unique_ys.shape[0]
            if n_ys == 1 and len(directions) > 0:
                # 悬停
                directions.append(directions[-1])
            elif n_ys > 2:
                # 需要至少三个不同y才能分辨悬停和垂直航线
                directions.append(np.pi / 2)
            else:
                directions.append(np.nan)
        else:
            if not robust:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[s], y[s])
            else:
                slope, intercept = stats.siegelslopes(y[s], x[s])
            directions.append(np.arctan(slope))

    return directions


def prune_by_std(directions) -> tuple:
    """筛选掉航段两端航向偏差较大的异常值（通常为转向带来的误差）

    Parameters
    ----------
    directions
        飞机航向角

    Returns
    -------
    sub_range
        返回筛选掉两端异常值后的区间
    """
    n_pts = len(directions)
    std = directions.std()
    center = np.percentile(directions, 50)

    deltas = abs(directions - center)
    is_normal = deltas <= 2 * std

    dl = np.argmax(is_normal)
    dr = np.argmax(is_normal[::-1])

    return dl, n_pts - dr


def segment_by_std(directions, stable_window_size) -> list[tuple]:
    """假设该航段由两个航段和一个连续的转弯组成，则寻找某个划分点将其分成两部分，且两部分为独立的两个航段，因此应方差相近

    Parameters
    ----------
    directions
        飞机航向角
    stable_window_size
        用于确定为航线的窗口测点数。

    Returns
    -------
    sub_segments
        如果可以切分，则切分为两段，并通过stable_window_size筛选掉过短的段。
        否则不分段直接返回 [(0, n_pts)]。
    """
    n_pts = len(directions)
    left, right = 0, n_pts
    lstd, rstd, mid = 100, -100, -1

    while right - left > 1:
        mid = (left + right) // 2
        lstd, rstd = directions[:mid].std(), directions[mid:].std()
        if lstd > rstd:
            right = mid
        else:
            left = mid

    if abs(lstd - rstd) < 0.1:
        ret = (0, mid), (mid, n_pts)
    else:
        ret = (0, n_pts),

    ret = [r for r in ret if r[1] - r[0] > stable_window_size]

    return ret


def segment_by_diff_directions(directions, stable_window_size, stable_direction_tol):
    diff_direction = np.r_[diff_directions(directions), 1e9]  # 保证最后一个航段也会统计进来

    segments = []
    first = None
    for i in range(len(diff_direction)):
        ddi = diff_direction[i]
        if abs(ddi) < stable_direction_tol:
            if first is None:
                first = i
        else:
            if first is not None:
                if i - first > stable_window_size:
                    segments.append((first, i))
                first = None

    new_segments = []
    for a, b in segments:
        ds = directions[a:b]
        if ds.std() > stable_direction_tol:
            new_segs = segment_by_std(ds, stable_window_size)
            new_segments.extend(apply_delta_to_segments(new_segs, a))
        else:
            new_segments.append((a, b))
    segments = new_segments

    new_segments = []
    for a, b in segments:
        new_segs = prune_by_std(directions[a:b])
        new_segments.extend(apply_delta_to_segments((new_segs,), a))
    segments = new_segments

    return segments


def split_lines(
        x, y,
        window_size=None,
        stable_window_size=None,
        window_length=10,
        stable_window_length=50,
        stable_direction_tol: float = 3,
):
    """通过滑动窗口求取区间航向，并依据航向的变化幅度分割为若干测线

    Parameters
    ----------
    x, y
        航线水平路径
    window_size, window_length
        用于分析航向的窗口测点数。应当大于飞机产生明显位移的窗口大小。
        过小时可能会导致测线划分对飞机飞行扰动敏感产生错误划分，过大时可能导致转弯等非平稳飞行测点也被容纳进测线。
    stable_window_size
        用于确定为航线的窗口测点数。应当大于转弯等非平稳飞行过程的窗口大小。
        过小时可能导致转弯等非平稳测点被统计为测线，过大时可能导致某些较短测线被意外移除。
    window_length, stable_window_length
        对应的窗口长度。若不指定对应size而指定length，则自动统计测点间距然后通过length求取窗口测点数。
    stable_direction_tol
        航向角偏差容差（单位 `度` ）

    Returns
    -------
    segments
        初步拆分后的若干航段
    """
    n_pts = len(x)

    if window_size is None or stable_window_size is None:
        skip = 2
        dx, dy = np.diff(x[::skip]), np.diff(y[::skip])
        d = np.sqrt(np.percentile(dx ** 2 + dy ** 2, 50)) / skip

        min_group_points = 5
        if window_size is None:
            window_size = max(int(np.ceil(window_length / d)), min_group_points)
        if stable_window_size is None:
            stable_window_size = max(int(np.ceil(stable_window_length / d)), min_group_points)

    x, y = check_slope(x, y)

    windows = [(idx, idx + window_size) for idx in range(n_pts - window_size)]
    directions = analyze_directions(x, y, windows, progress=True)
    directions = np.rad2deg(directions)

    segments = segment_by_diff_directions(directions, stable_window_size, stable_direction_tol)
    segments = [(first, last + window_size + 1) for first, last in segments]

    return segments


def analyze_lines(x, y, segments):
    details = analyze_lines_details(x, y, segments)
    return details.slope, details.intercept


def analyze_lines_details(x, y, segments):
    x, y = check_slope(x, y)

    ret = []
    for first, last in segments:
        sx = x[first:last]
        sy = y[first:last]
        result = stats.linregress(sx, sy)
        ret.append(result)

    return pd.DataFrame(ret)


def remove_aux_flights(x, y, segments, stable_threshold: float = 6):
    directions = analyze_directions(x, y, segments, robust=True)
    directions = np.rad2deg(directions)
    weights = [s[1] - s[0] for s in segments]

    d_dir = directions - directions[0]
    d_dir = regulate_delta_directions(d_dir, inplace=True)

    selected_direction = weighted_percentile(d_dir, weights, 50) + directions[0]
    selected_direction = regulate_delta_directions(selected_direction)

    angle_offsets = abs(regulate_delta_directions(directions - selected_direction))
    condition = angle_offsets < stable_threshold

    aux_removed = []

    for idx in np.where(condition)[0]:
        aux_removed.append(segments[idx])

    return aux_removed
