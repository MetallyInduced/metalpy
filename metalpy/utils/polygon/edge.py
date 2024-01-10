import numpy as np


class Edge:
    __slots__ = ['p1', 'p2', 'ymin', 'ymax', 'xmin', 'xmax', 'k', 'b']

    def __init__(self, p1, p2):
        self.p1, self.p2 = p1, p2
        self.ymin = min(p1[1], p2[1])
        self.ymax = max(p1[1], p2[1])
        self.xmin = min(p1[0], p2[0])
        self.xmax = max(p1[0], p2[0])

        if p1[0] != p2[0]:  # 非垂直于x轴
            dx, dy = p2 - p1
            self.k = dy / dx
            self.b = p1[1] - self.k * p1[0]
        else:
            self.k = np.inf
            self.b = p1[0]

    @property
    def xs(self):
        return np.c_[self.p1[0], self.p2[0]]

    @property
    def ys(self):
        return np.c_[self.p1[1], self.p2[1]]

    @property
    def pts(self):
        return np.vstack([self.p1, self.p2])

    @property
    def signed_theta(self):
        return np.arctan2(*(self.p2 - self.p1)[::-1])

    @property
    def length(self):
        return np.linalg.norm(self.p2 - self.p1)

    def contains(self, x=None, y=None, closed=True):
        """判断线段中是否包含指定点

        Parameters
        ----------
        x, y
            指定点坐标，如果只指定其中之一，则只判断该轴方向是否在线段上
        closed
            是否包含端点

        Returns
        -------
        flag
            指定点或点序列是否在线段上
        """
        if closed:
            def cmp(a, b): return a <= b
        else:
            def cmp(a, b): return a < b

        if x is not None:
            fx = cmp(self.xmin, x) & cmp(x, self.xmax)
        else:
            fx = True
        if y is not None:
            fy = cmp(self.ymin, y) & cmp(y, self.ymax)
        else:
            fy = True

        return fx & fy

    def intersects(self, pts):
        x, y = pts[:, 0], pts[:, 1]
        if np.isinf(self.k):
            # 处理与垂直线相交的情形
            # 由于是多面体，因此与该直线相交的射线一定与其重合
            # 其所在直线会与该边的相邻两个边也存在交点
            # 交于下端点视为相交，其余位置不视为相交
            # 可以保证边缘点被视为多边形内
            # 1. 点p在该线段之下，则只会与相邻边有2个交点
            # 2. 点p在该线段与下方邻边交点，则会与相邻边和该边有3个交点
            # 3. 点p在该线段中间及上方邻边交点，则只会与上方邻边有1个交点
            indices = x == self.b
            indices[indices] = y[indices] == self.ymin
            return indices
        elif self.k == 0:
            return self.contains(x=x) & (y <= self.ymin)
        else:
            y_intersection = self.k * x + self.b
            return self.contains(y=y_intersection) & (y <= y_intersection)

    def intersects_edge(self, edge: 'Edge', closed=False):
        if self.k == edge.k:
            if self.b == edge.b:
                # 重合线，调整某条线的端点
                if np.isinf(self.k):
                    m = self.contains(edge.ys, closed=closed)
                    if np.any(m):
                        return edge.pts[np.where(m)[0][0]]
                else:
                    m = edge.contains(self.xs, closed=closed)
                    if np.any(m):
                        return self.pts[np.where(m)[0][0]]
            return None
        else:
            y = None
            if np.isinf(self.k):
                x = self.b
                if edge.contains(self.b, closed=closed):
                    y = edge.k * x + edge.b
                    if not self.contains(y=y, closed=closed):
                        y = None
            elif np.isinf(edge.k):
                x = edge.b
                if self.contains(edge.b, closed=closed):
                    y = self.k * x + self.b
                    if not edge.contains(y=y, closed=closed):
                        y = None
            else:
                x = (edge.b - self.b) / (self.k - edge.k)
                if self.contains(x, closed=closed) and edge.contains(x, closed=closed):
                    y = self.k * x + self.b

            if y is not None:
                return x, y
            else:
                return None
