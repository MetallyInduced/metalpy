import warnings

import numpy as np

from . import Edge
from metalpy.utils.numpy import get_resolution


def re_intersect(pts, verbose=True):
    """基于重相交算法调整复杂多边形，并转换为适用于耳切法的简单多边形

    将多条边的内交点拆分为若干个新顶点，分别连接多个子连通多边形，
    从而消除自交边，保证多边形内部最多只存在重合顶点，从而解决自相交对三角化的影像

    Parameters
    ----------
    pts
        顺序指定的多边形顶点，并且首尾相连
    verbose
        是否输出辅助信息，包括顶点过多时显示进度条

    Returns
    -------
    pts
        重相交算法计算后的多边形顶点

    Notes
    -----
    该算法目前最多只支持 2 或 3 条边自相交，更多条边的自相交拆分结果可能不正确
    """
    pts = np.asarray(pts)
    splits = check_inner_intersection(pts)

    n_splits = len(splits)
    rng = range(n_splits)

    if n_splits > 1000 and verbose:
        import tqdm
        rng = tqdm.tqdm(rng)

    for k in rng:
        edges, intersection = splits[k]
        if len(edges) == 2:
            pts = disassemble_2(*edges, intersection, pts, splits)
        else:
            pts = disassemble_n(edges, intersection, pts, splits)

    return pts


def extract_strongly_connected_components(intersections):
    involved_edges = set()
    selected_intersections = set()
    adjacent_edges: dict[int, set] = {}

    for i, j in intersections:
        roots = adjacent_edges.setdefault(i, set()) & adjacent_edges.setdefault(j, set())
        if len(roots) > 0:
            # 两条边的端点都已经被连接，为强联通
            selected_intersections.add((i, j))
            involved_edges.add(i)
            involved_edges.add(j)

            for r in roots:
                selected_intersections.add((min(r, i), max(r, i)))
                selected_intersections.add((min(r, j), max(r, j)))
                involved_edges.add(r)

        adjacent_edges.setdefault(i, set()).add(j)
        adjacent_edges.setdefault(j, set()).add(i)

    return selected_intersections, involved_edges


def approx_multi_intersection(splits):
    if len(splits) == 0:
        return []

    dtype = splits[0][-1].dtype
    atol = get_resolution(dtype)
    rtol = np.sqrt(atol)

    intersections = np.empty([0, 2], dtype=dtype)
    merged_splits = []
    for (i, j), inter in splits:
        mask = np.all(np.isclose(inter, intersections, atol=atol, rtol=rtol), axis=1)
        if np.any(mask):
            merged_splits[np.where(mask)[0][0]][1].append((i, j))
        else:
            merged_splits.append([inter, [(i, j)]])
            intersections = np.vstack([intersections, inter])

    return merged_splits


def check_multi_intersection(edges, splits):
    merged_splits = approx_multi_intersection(splits)

    checked_merged_splits = []
    for intersection, old_edges in merged_splits:
        checked = []

        if len(old_edges) == 1:
            checked.append([old_edges[0], intersection])
        else:
            if len(old_edges) >= 3:
                selection, involved = extract_strongly_connected_components(old_edges)
                checked.append((sorted(involved), intersection))
                old_edges = set(old_edges) - selection

            if len(old_edges) != 0:
                for i, j in old_edges:
                    # 非强连通分量的成员，重新计算交点
                    # approx_multi_intersection中将他们交点统合为一个了
                    checked.append(([i, j], edges[i].intersects_edge(edges[j])))

        checked_merged_splits.extend(checked)

    return checked_merged_splits


def check_inner_intersection(pts):
    n_edges = len(pts)
    edges = []
    for i in range(n_edges):
        edges.append(Edge(pts[i], pts[(i + 1) % n_edges]))

    splits = []
    for i in range(n_edges):
        # 注意需要排除掉邻边
        # i == 0 时还需要排除左邻边（即 j == n_edges - 1）
        for j in range(i + 2, n_edges - (i == 0)):
            intersection = edges[i].intersects_edge(edges[j])
            if intersection is not None:
                splits.append(((i, j), np.asarray(intersection)))

    merged = check_multi_intersection(edges, splits)

    return merged


def disassemble_n(edges, intersection, pts, splits):
    if len(edges) > 3:
        # TODO: 考虑多于3条边的自相交如何处理
        warnings.warn(f'Only 2- or 3-edge self-intersection is supported.'
                      f' Got {len(edges)}, which is likely to yield incorrect results.')

    n_pts = len(pts)
    n_edges = len(edges)

    intersecting_edges = np.c_[np.roll(edges, 1), edges]

    si, sj = edges[0], edges[-1]
    polys = [[[intersection], pts[sj + 1:], pts[:si + 1]]]

    flag, multiplier = 1, 1 if len(edges) % 2 else -1
    flags = np.logspace(0, n_edges - 1, n_edges, base=multiplier, dtype=int)

    for (i, j), flag in zip(intersecting_edges[1:], flags[1:]):
        if flag == 1:
            polys.append([[intersection], pts[i + 1:j + 1]])
        else:
            polys.append([[intersection], pts[j:i:-1]])

    n_polys = [sum(len(pp) for pp in poly) for poly in polys]
    bases = np.cumsum([0, *n_polys[:-1]])

    # 交点拆分后，原本的一对相交线段各自都被拆分为两个线段
    breaking_edges = [[] for _ in range(n_edges)]
    for i in range(n_edges):
        prev = (i - 1) % n_edges
        base, flag = bases[i], flags[i]
        if flag == 1:
            breaking_edges[i].append(base + n_polys[i] - 1)
            breaking_edges[prev].append(base)
        else:
            breaking_edges[i].append(base)
            breaking_edges[prev].append(base + n_polys[i] - 1)

    pts = np.vstack(sum(polys, []))

    for p in range(len(splits)):
        edges_, intersection_ = splits[p]

        def _check_index(idx):
            for (i, j), base, flag in zip(intersecting_edges, bases, flags):
                if i > j:
                    if j < idx <= i:
                        continue
                else:
                    if not (i < idx <= j):
                        continue

                # base 即为之前已有的顶点数
                # +1 为该分区内补充的新交点 `[intersection]`
                if flag == 1:
                    idx = (idx - (i + 1)) % n_pts + (base + 1)
                else:
                    idx = j - idx + (base + 1) - 1  # -1源自方向反转，边的起点发生变化

                break

            n_new_pts = len(pts)
            temp_edge = Edge(pts[idx], pts[(idx + 1) % n_new_pts])
            for vs in breaking_edges:
                if idx in vs and not temp_edge.contains(*intersection_):
                    idx = sum(vs) - idx
                    break

            return idx

        edges_ = sorted([_check_index(i) for i in edges_])
        splits[p] = (edges_, intersection_)

    return pts


def disassemble_2(i, j, intersection, pts, splits):
    n_pts = len(pts)

    poly1 = [[intersection], pts[j + 1:], pts[:i + 1]]  # A - j+1 - i ，方向不变
    poly2 = [[intersection], pts[j:i:-1]]  # A_ - j - i+1 ，方向反转

    pts = np.vstack(poly1 + poly2)

    n_poly1 = sum([len(a) for a in poly1])

    for p in range(len(splits)):
        edges_, intersection_ = splits[p]

        def _check_index(idx):
            # 重置下标
            if idx > j or idx <= i:
                idx = (idx - j - 1) % n_pts + 1
            else:
                idx = j - idx + n_poly1 + 1 - 1

            n_new_pts = len(pts)
            temp_edge = Edge(pts[idx], pts[(idx + 1) % n_new_pts])
            breaking_vertices = [
                [0, n_poly1],  # A - j+1 和 A_ - j
                [n_new_pts - 1, n_poly1 - 1]  # i+1 - A 和 i - A_
            ]  # 交点拆分后，原本的一对相交线段各自都被拆分为两个线段，需要判断交点是在二者中的哪一个
            for vs in breaking_vertices:
                if idx in vs and not temp_edge.contains(*intersection_):
                    idx = sum(vs) - idx
                    break

            return idx

        edges_ = sorted([_check_index(i) for i in edges_])
        splits[p] = (edges_, intersection_)

    return pts


def plot_disassembled_polys(polys, ax=None):
    from metalpy.utils.matplotlib import check_axis

    with check_axis(ax, figsize=(5, 5)) as ax:
        for p in polys:
            p = np.vstack(p)
            closed = np.r_[p, [p[0]]]
            ax.plot(*closed.T)
            for i in range(len(p)):
                start, end = p[i], p[(i + 1) % len(p)]
                delta = end - start
                ax.arrow(*(start + delta * 0.49), *delta * 0.02, width=0.003)

        pts = np.vstack(sum(polys, []))
        for ii in range(len(pts)):
            ax.text(*pts[ii], str(ii))
        for p in pts:
            ax.scatter(*p, s=0.3)
