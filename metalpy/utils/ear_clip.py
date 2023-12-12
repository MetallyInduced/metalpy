from __future__ import annotations

import warnings
from typing import Literal, Iterable

import numpy as np
from numpy.linalg import norm


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
        self.angle = 0
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
        self.triangle = self.polygon.extract_triangle(self.index)
        a = self.triangle[1] - self.triangle[0]
        b = self.triangle[2] - self.triangle[1]
        self.angle = cross(a, b) / norm(a) / norm(b)

        self.invalidate()

    @property
    def reflex(self):
        # 判断是否为内凹顶点
        return self.polygon.winding * self.angle < 0

    @property
    def vertex(self):
        return self.triangle[1]

    @property
    def prev_index(self):
        return self.polygon.prev(self.index)

    @property
    def next_index(self):
        return self.polygon.next(self.index)

    def clip(self):
        self.polygon.remove(self.index)
        return self.prev.index, self.index, self.next.index

    def remove(self):
        self.prev.next = self.next
        self.next.prev = self.prev

    def invalidate(self):
        self.conflicted_vertex = None

    def test_is_ear_many(self, vertices: Iterable['VertexTriangle']):
        if self.conflicted_vertex in vertices:
            return False

        for v in vertices:
            if not self.test_inside(v):
                return False

        return True

    def test_inside(self, vertex: 'VertexTriangle'):
        if vertex in [self, self.prev, self.next]:
            return True  # 顶点不认为在三角形内
        if is_inside_convex_polygon(vertex.vertex, self.triangle):
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
    return np.sign(cross(la, lb))


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
    # vectors为p到所有顶点的向量
    vectors = pts - p

    view = np.lib.stride_tricks.sliding_window_view(
        np.r_[vectors, vectors[:1]], (2, 2)
    ).squeeze()

    # w由外积得到的，代表p到每个顶点v的向量旋转到下一个顶点上的时钟序
    # +1顺时针，-1逆时针，0共线
    w = determine_line_winding(view[:, 0].T, view[:, 1].T)

    if np.allclose(w, 0):
        # 多边形自身共线且与p共线
        # 判断p是否在任意线段上
        flag = vectors * np.roll(vectors, -1, axis=0) <= 0
        return np.any(np.all(flag, axis=1))

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
        如果遇到非简单多边形，是否抛出异常

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

        if vert.reflex:
            reflex_vertices.add(vert)

    rng = range(n_vertices - 2)

    if n_vertices > 200 and verbose:
        # TODO: 用c++重写？
        import tqdm

        warnings.warn(f'Applying ear-clipping on polygons'
                      f' with more than 200 vertices may lead to performance issue.'
                      f' (n_vertices = {n_vertices})'
                      f'\nProgress bar is automatically enabled,'
                      f' disable it by setting `verbose` = False.')
        rng = tqdm.tqdm(rng)

    for i in rng:
        best_to_remove: VertexTriangle | None = None
        for v in vertices:
            v: VertexTriangle

            if v.reflex:
                continue

            if v.test_is_ear_many(reflex_vertices):
                if best_to_remove is None:
                    best_to_remove = v
                    break

        if best_to_remove is None:
            # something wrong
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

        if best_to_remove.reflex:
            reflex_vertices.remove(best_to_remove)

        skip_reflex = len(reflex_vertices) == 0
        if prev_.reflex:
            reflex_vertices.add(prev_)
        elif not skip_reflex:
            try:
                reflex_vertices.remove(prev_)
            except KeyError:
                pass
            else:
                pass

        if next_.reflex:
            reflex_vertices.add(next_)
        elif not skip_reflex:
            try:
                reflex_vertices.remove(next_)
            except KeyError:
                pass
            else:
                pass

    return triangles
