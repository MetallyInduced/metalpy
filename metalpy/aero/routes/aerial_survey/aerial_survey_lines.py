from __future__ import annotations

import copy
from collections import namedtuple
from typing import Sequence, Generator, overload, Any

import numpy as np

from metalpy.aero.utils.line_analysis import diff_directions, remove_aux_flights_by_directions
from metalpy.carto.coords import Coordinates
from metalpy.utils.batch import Batch
from metalpy.utils.matplotlib import check_axis
from .aerial_survey import AerialSurvey
from .aerial_survey_line import AerialSurveyLine


class AerialSurveyLines:
    LineSpec = namedtuple('LineSpec', ['direction', 'line_interval'])

    def __init__(self, lines: Sequence[AerialSurveyLine]):
        """构造航空探测测线对象

        Parameters
        ----------
        lines
            测线信息
        """
        self.lines = np.fromiter(lines, dtype=object)

    @property
    def length(self):
        return sum(line.length for line in self)

    def pop_auxiliary(self, direction_threshold: float = 6):
        factors = [line.get_line_spec(check_std=False)[:2] for line in self]
        directions = [np.arctan2(-a, b) for a, b in factors]
        weights = [len(line) for line in self]

        kept_idx = remove_aux_flights_by_directions(directions, weights, stable_threshold=direction_threshold)
        ptr = 0

        kept = []
        popped = []
        for i in range(len(self)):
            line_i = self.lines[i]
            if i != kept_idx[ptr]:
                popped.append(line_i)
            else:
                kept.append(line_i)
                ptr += 1

                if ptr >= len(kept_idx):
                    # 剩下的全部被视为辅助航线，加入排除列表
                    for j in range(i + 1, len(self)):
                        popped.append(self.lines[j])
                    break

        self.lines = kept
        return AerialSurveyLines(popped)

    def remove_auxiliary(self, direction_threshold: float = 6, inplace=False):
        if inplace:
            ret = self
        else:
            ret = copy.copy(self)

        ret.pop_auxiliary(direction_threshold=direction_threshold)
        return ret

    def merge_neighbors(self, inplace=False):
        directions, intercepts, cos_thetas = self.get_detailed_line_specs()

        di = abs(np.diff(intercepts)) * cos_thetas[:-1]
        criteria = np.percentile(di, 40) / 2

        merged_lines = [[self.lines[0]]]
        for i, merged in enumerate(di < criteria):
            line = self.lines[i + 1]
            if merged:
                merged_lines[-1].append(line)
            else:
                merged_lines.append([line])

        merged_lines = [AerialSurveyLine.concat(lines) for lines in merged_lines]

        if inplace:
            self.lines = merged_lines
            ret = self
        else:
            ret = AerialSurveyLines(merged_lines)

        return ret

    @overload
    def __getitem__(self, item: int) -> AerialSurveyLine: ...

    @overload
    def __getitem__(self, item: Any) -> AerialSurveyLines: ...

    def __getitem__(self, item) -> AerialSurveyLine | AerialSurveyLines:
        ret = self.lines[item]
        if np.ndim(ret) == 0:
            return ret
        else:
            return AerialSurveyLines(ret)

    def __len__(self):
        return len(self.lines)

    def __iter__(self) -> Generator[AerialSurveyLine]:
        yield from self.lines

    def merge(self):
        return AerialSurveyLine.concat(self.lines)

    def to_survey(self):
        ret = self.merge()
        return AerialSurvey(ret.position, ret.data)

    def get_line_specs(self, group_direction_tol=10):
        directions, intercepts, cos_thetas = self.get_detailed_line_specs()
        directions = np.rad2deg(directions)

        splits = np.where(abs(diff_directions(directions)) > group_direction_tol)[0] + 1
        directions = [ds.mean() for ds in np.array_split(directions, splits)]
        cos_thetas = [ct.mean() for ct in np.array_split(cos_thetas, splits)]
        intervals = [
            np.abs(np.diff(ints) * ct).mean()
            for ints, ct in zip(np.array_split(intercepts, splits), cos_thetas)
        ]

        return [
            AerialSurveyLines.LineSpec(d, i)
            for d, i in zip(directions, intervals)
        ]

    def get_detailed_line_specs(self):
        """获取详细的航线分析信息

        Returns
        -------
        directions, intercepts, cos_thetas
            每条航线的朝向（-pi/2~pi/2），每条航线到最近坐标轴的截距，每条航线到最近坐标轴的余弦角
        """
        lines = Batch.of(self.lines)
        line_specs = np.asarray(lines.get_line_spec())
        centers = np.asarray(lines.center)
        rx, ry = centers.mean(axis=0)  # 参考原点

        directions = []
        intercepts = []
        slopes = []
        for dx, dy, intercept in line_specs:
            if dy == -1:
                directions.append(np.arctan(dx))
                intercepts.append(dx * rx + intercept)
                slopes.append(dx)
            else:
                # 注意其定义域为[(1/2) * pi, (3/2) * pi]
                # 需要进行转换
                directions.append(np.pi / 2 - np.arctan(dy))
                intercepts.append(dy * ry + intercept)
                slopes.append(dy)

        directions = np.asarray(directions)
        intercepts = np.asarray(intercepts)
        slopes = np.asarray(slopes)
        directions[directions > np.pi / 2] -= np.pi

        cos_thetas = 1 / np.sqrt(1 + slopes ** 2)

        return directions, intercepts, cos_thetas

    def plot(self, ax=None, show=True, fontsize=10, **kwargs):
        with check_axis(ax, show=show, figsize=(5, 5)) as ax:
            for i, line in enumerate(self.lines):
                pos = np.asarray(line.position)
                ax.plot(*pos.T, **kwargs)
                ax.text(*Coordinates(pos).bounds.center, str(i), fontsize=fontsize)

    def __copy__(self):
        return AerialSurveyLines(self.lines)
