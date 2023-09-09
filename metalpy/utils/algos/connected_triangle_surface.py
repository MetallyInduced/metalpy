from functools import reduce


class ConnectedTriangleSurfaces:
    def __init__(self):
        """用于依据联通性（是否有相邻边）将多个三角面的顶点分组
        """
        self.point_groups: list[set[int]] = []

    def add(self, pts):
        vs = set(pts)
        candidates = []
        diffs = []

        # 考虑到面在各个局部体内的连续性，一般而言新的面会邻接最后一个点集或者属于新的点集合
        for group in reversed(self.point_groups):
            diff = vs.difference(group)
            if len(diff) <= 1:
                candidates.append(group)
                diffs.append(diff)

                if len(candidates) > 1:
                    break  # 因为针对三角面，所以一定只能同时邻接两个组

        n_adjacent = len(candidates)

        if n_adjacent == 0:
            target_group = set(pts)
            self.point_groups.append(target_group)
        elif n_adjacent == 1:
            assert len(diffs) == 1
            candidates[0].update(diffs.pop())
        else:
            for g in candidates[1:]:
                self.point_groups.remove(g)

            reduce(lambda x, y: x.update(y), candidates)

    def get_groups(self):
        return self.point_groups
