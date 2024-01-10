from __future__ import annotations

import warnings
from typing import Literal, Iterable, Collection

import numpy as np

from metalpy.utils.numpy import get_resolution


class ClippablePolygon:
    def __init__(self, pts):
        self.winding = determine_winding(pts)
        self.pts = pts
        self.nexts = np.arange(1, pts.shape[0] + 1, dtype=np.int64)
        self.nexts[-1] = 0
        self.prevs = np.arange(-1, pts.shape[0] - 1, dtype=np.int64)
        self.prevs[0] = pts.shape[0] - 1

    def remove(self, index):
        n = self.next(index)
        p = self.prev(index)
        self.prevs[n] = p
        self.nexts[p] = n

    def regulate_index(self, index):
        if index < 0:
            index += self.pts.shape[0]
        elif index >= self.pts.shape[0]:
            index -= self.pts.shape[0]
        return index

    def prev(self, index):
        index = self.regulate_index(index)
        return self.prevs[index]

    def next(self, index):
        index = self.regulate_index(index)
        return self.nexts[index]

    def extract_triangle(self, index):
        indices = [self.prev(index), index, self.next(index)]
        return self.pts[indices]

    def check_index(self, index):
        return self.prev(self.next(index)) == index


class VertexTriangle:
    def __init__(self, index, polygon: ClippablePolygon):
        self.index = index
        self.polygon = polygon
        self.triangle = None
        self.edge_angles = None
        self.angle = 0
        self.area = 0
        self.hanging = False
        self.conflicted_vertex: VertexTriangle | Literal[False] | None = None

        self.prev = self
        self.next = self

        self.assemble_triangle()

    def append(self, node: 'VertexTriangle'):
        last = self.prev
        last.next = node
        node.prev = last

        self.prev = node
        node.next = self

    def assemble_triangle(self):
        self.invalidate()

        self.triangle = self.polygon.extract_triangle(self.index)

        a, b, c = edge_vectors = np.roll(self.triangle, -1, axis=0) - self.triangle

        area_2 = cross(a, b)
        len_a, len_b, len_c = lengths = np.linalg.norm(edge_vectors, axis=1)

        if np.any(lengths == 0):
            self.angle = 0
            self.area = 0
            self.hanging = True
        else:
            self.angle = area_2 / len_a / len_b
            self.area = area_2 / 2

            if abs(self.angle) < 1e-14:
                self.angle = 0
                self.area = 0

        edge_angles = [
            check_slope(self.triangle[0], self.triangle[1]),
            check_slope(self.triangle[0], self.triangle[2]),
            check_slope(self.triangle[1], self.triangle[2]),
        ]
        self.edge_angles = np.asarray([
            [edge_angles[1], edge_angles[0]],
            [check_angle(edge_angles[0] + np.pi), edge_angles[2]],
            [check_angle(edge_angles[2] + np.pi), check_angle(edge_angles[1] + np.pi)],
        ])

    @property
    def reflex(self):
        # 判断是否为内凹顶点
        return self.polygon.winding * self.angle < 0

    @property
    def conflictual(self):
        # 判断是否为冲突顶点(即可能与凸顶点发生冲突)
        # 包括内凹顶点和0面积顶点
        return self.reflex or self.angle == 0

    @property
    def clippable(self):
        # 判断是否为内凹顶点
        return not self.reflex or self.angle == 0

    @property
    def vertex(self):
        return self.triangle[1]

    @property
    def prev_index(self):
        return self.polygon.prev(self.index)

    @property
    def next_index(self):
        return self.polygon.next(self.index)

    @property
    def vertex_edge_angles(self):
        return self.edge_angles[1]

    def clip(self):
        self.polygon.remove(self.index)
        return self.prev.index, self.index, self.next.index

    def remove(self):
        self.prev.next = self.next
        self.next.prev = self.prev

    def invalidate(self):
        self.hanging = False
        self.conflicted_vertex = None

    def test_is_ear_many(self, vertices: Iterable['VertexTriangle']):
        if self.angle == 0:
            return self.hanging

        if self.conflicted_vertex in vertices:
            return False

        for v in vertices:
            if not self.test_inside(v):
                return False

        return True

    def test_inside(self, vertex: 'VertexTriangle'):
        if vertex in [self, self.prev, self.next]:
            return True  # 顶点不认为在三角形内

        coinciding_idx = _coinciding_vertex(vertex.vertex, self.triangle)
        if coinciding_idx >= 0:
            if _is_edges_overlapping(self.edge_angles[coinciding_idx], vertex.vertex_edge_angles):
                self.conflicted_vertex = vertex
                return False
            else:
                return True
        elif is_inside_convex_polygon(vertex.vertex, self.triangle):
            self.conflicted_vertex = vertex
            return False
        else:
            return True

    def __hash__(self):
        return hash(self.index)

    def __iter__(self):
        ptr = self
        yield ptr
        ptr = ptr.next
        while ptr != self:
            yield ptr
            ptr = ptr.next

    def plot(self, focus=None, ax=None, index=False, vertex=False):
        from metalpy.utils.matplotlib import check_axis

        with check_axis(ax, figsize=(5, 5)) as ax:
            for v in self:
                ax.plot(*v.triangle.T, c='grey')
            if focus is not None:
                if not isinstance(focus, dict):
                    focus = {'red': focus}
                for c, v in focus.items():
                    ax.plot(*v.triangle.T, c=c)
                    if vertex:
                        ax.scatter(*v.triangle.T, s=3, c=c)
            if index:
                for v in self:
                    if isinstance(index, Collection):
                        if v.index not in index and v not in index:
                            continue
                        ax.scatter(*v.vertex, s=2, c='orange')
                    ax.text(*v.vertex, str(v.index))
            if vertex:
                for v in self:
                    ax.scatter(*v.vertex, s=1, c='blue')

    def plot_directions(self, focus=None, ax=None, index=False, vertex=False):
        from metalpy.utils.matplotlib import check_axis

        with check_axis(ax, figsize=(5, 5)) as ax:
            from metalpy.utils.polygon.re_intersect import plot_disassembled_polys

            pts = np.asarray([v.vertex for v in self])
            plot_disassembled_polys([[pts]], ax=ax)

            self.plot(focus=focus, index=index, vertex=vertex, ax=ax)

    def __repr__(self):
        reflex = 'Reflex' if self.reflex else ''
        return f'{reflex}Vertex[{self.index}, {self.angle:.3e}, {self.prev_index} -> {self.index} -> {self.next_index}]'

    def __getitem__(self, idx):
        for v in self:
            if v.index == idx:
                return v
        raise IndexError(f'Vertex[{idx}] not found in vertices.')


def segment_area_times_2(x2, x1, y2, y1):
    return (x2 - x1) * (y2 + y1)


def determine_winding(pts):
    """计算给定有序二维点集的时钟序

    Parameters
    ----------
    pts
        有序二维点集

    Returns
    -------
    ret
        -1代表顺时针方向，1代表逆时针方向
    """
    v = np.sum(segment_area_times_2(pts[1:, 0], pts[:-1, 0], pts[1:, 1], pts[:-1, 1]))
    v += segment_area_times_2(pts[0, 0], pts[-1, 0], pts[0, 1], pts[-1, 1])
    return -np.sign(v)


def cross(la, lb):
    return la[0] * lb[1] - la[1] * lb[0]


def determine_line_winding(la, lb) -> int:
    atol = get_resolution(la)

    rot = cross(la, lb)
    rot[abs(rot) < atol] = 0

    return np.sign(rot)


def check_slope(p0, p1):
    dx, dy = p1 - p0

    return check_angle(np.arctan2(dy, dx))


def check_angle(a, atol=1e-15):
    if abs(a) < atol:
        a = 0

    while a < 0:
        a += np.pi * 2

    while a > np.pi * 2:
        a -= np.pi * 2

    return a


def _coinciding_vertex(p, pts) -> bool:
    idx = -1
    mask = np.all(pts == p, axis=1)

    if np.any(mask):
        idx = np.where(mask)[0][0]

    return idx


def _is_edges_overlapping(a, b) -> bool:
    atol = get_resolution(a.dtype) * 100
    atol = rtol = np.sqrt(atol)

    a0, a1 = a
    b0, b1 = b

    a1 = check_angle(a1 - a0, atol=atol)
    b0 = check_angle(b0 - a0, atol=atol)
    b1 = check_angle(b1 - a0, atol=atol)
    a0 = 0.0

    if a1 < np.pi:
        fb0 = b0 < a1
        fb1 = b1 < a1
    else:
        fb0 = b0 > a1
        fb1 = b1 > a1

    fob0 = np.any(np.isclose([a0, a1], b0, atol=atol, rtol=rtol))
    fob1 = np.any(np.isclose([a0, a1], b1, atol=atol, rtol=rtol))

    fb0 = fb0 and not fob0
    fb1 = fb1 and not fob1

    return fb0 or fb1 or fob0 and fob1


def is_inside_convex_polygon(p, pts) -> bool:
    """判断给定点p是否存在于凸多边形中

    Parameters
    ----------
    p
        给定点p
    pts
        凸多边形顶点

    Returns
    -------
    ret
        p是否存在于凸多边形中
    """
    bmin, bmax = np.min(pts, axis=0), np.max(pts, axis=0)
    if np.any(p < bmin) or np.any(p > bmax):
        # 粗判，p不在凸多边形的外接矩形内
        return False

    # vectors为p到所有顶点的向量
    vectors = pts - p

    view = np.lib.stride_tricks.sliding_window_view(
        np.r_[vectors, vectors[:1]], (2, 2)
    ).squeeze()

    # w由外积得到的，代表p到每个顶点v的向量旋转到下一个顶点上的时钟序
    # +1顺时针，-1逆时针，0共线
    w = determine_line_winding(view[:, 0].T, view[:, 1].T)

    # 如果多次旋转的方向相同（共线忽略），则代表目标在多边形内
    return np.all(w <= 0) or np.all(w >= 0)


def ear_clip(pts, verbose=True, check_non_simple=True):
    """实现基于耳切法的多边形三角化

    Parameters
    ----------
    pts
        多边形点集
    verbose
        是否输出辅助信息，包括顶点过多警告和顶点过多时显示进度条
    check_non_simple
        是否检查非简单多边形。为 `True` 时遇到非简单多边形会抛出异常，否则会直接返回部分三角化的结果（结果大概率错误）

    Returns
    -------
    ret
        array[len(pts) - 2, 3]，
        三角化后的面集合，每行指定一个三角形的三个顶点在pts中的下标

    Notes
    -----
        只支持简单多边形，即任意两条边的交点必须是顶点
    """
    n_vertices = pts.shape[0]
    polygon = ClippablePolygon(pts)
    vertices: VertexTriangle | None = None
    reflex_vertices: set[VertexTriangle] = set()
    triangles = np.empty((n_vertices - 2, 3), dtype=np.int64)

    for i in range(pts.shape[0]):
        vert = VertexTriangle(i, polygon)
        if vertices is None:
            vertices = vert
        else:
            vertices.append(vert)

        if vert.conflictual:
            reflex_vertices.add(vert)

    n_triangles = n_vertices - 2
    rng = range(n_triangles)

    if n_vertices > 200 and verbose:
        # TODO: 用c++重写？
        import tqdm

        warnings.warn(f'Applying ear-clipping on polygons'
                      f' with more than 200 vertices may lead to performance issue.'
                      f' (n_vertices = {n_vertices})'
                      f'\nProgress bar is automatically enabled,'
                      f' disable it by setting `verbose` to `False`.')
        rng = tqdm.tqdm(rng)

    for i in rng:
        candidates = sorted(
            (x for x in vertices if x.clippable),
            key=lambda x: abs(x.angle) if x.angle != 0 else 100,
            reverse=True
        )  # TODO: 使用堆优化，规避冗余的排序

        for v in candidates:
            if v.test_is_ear_many(reflex_vertices):
                best_to_remove = v
                break
        else:
            # 未能找到可剪切的凸顶点
            error_info = f'Non-simple polygon is not supported yet.'
            if check_non_simple:
                raise RuntimeError(
                    error_info +
                    ' Disable checks by setting `check_non_simple` to False to obtain partial result.'
                )
            else:
                if verbose:
                    warnings.warn(
                        error_info +
                        ' A partial result is returned and is very likely not correct.'
                    )
                return triangles[:i]

        triangles[i] = best_to_remove.clip()  # 从polygon中删除顶点

        prev_ = best_to_remove.prev
        next_ = best_to_remove.next

        best_to_remove.remove()
        if vertices == best_to_remove:
            vertices = next_

        prev_.assemble_triangle()
        next_.assemble_triangle()

        if best_to_remove.conflictual:
            reflex_vertices.remove(best_to_remove)

        skip_reflex = len(reflex_vertices) == 0
        if prev_.conflictual:
            reflex_vertices.add(prev_)
        elif not skip_reflex:
            try:
                reflex_vertices.remove(prev_)
            except KeyError:
                pass
            else:
                pass

        if next_.conflictual:
            reflex_vertices.add(next_)
        elif not skip_reflex:
            try:
                reflex_vertices.remove(next_)
            except KeyError:
                pass
            else:
                pass

    return triangles
