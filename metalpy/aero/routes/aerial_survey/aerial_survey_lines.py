from __future__ import annotations

from collections import namedtuple
from typing import Sequence, Generator

import numpy as np

from metalpy.aero.utils.line_analysis import diff_directions
from metalpy.carto.coords import Coordinates
from metalpy.utils.batch import Batch
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
        self.lines = np.asarray(lines)

    def __getitem__(self, item):
        return AerialSurveyLines(self.lines[item])

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
        from matplotlib import pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = None

        for i, line in enumerate(self.lines):
            pos = np.asarray(line.position)
            ax.plot(*pos.T, **kwargs)
            ax.text(*Coordinates(pos).bounds.center, str(i), fontsize=fontsize)

        if show and fig is not None:
            fig.show()
