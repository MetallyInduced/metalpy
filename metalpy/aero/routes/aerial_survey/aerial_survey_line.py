from __future__ import annotations

from typing import overload, Collection

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from metalpy.aero.utils.line_analysis import check_slope
from metalpy.utils.array_ops import get_column, cstack, get_row
from . import AerialSurvey, AerialSurveyPoint


class AerialSurveyLine(AerialSurvey):
    def __init__(self, position, data=None):
        super().__init__(position, data)

    def get_line_spec(self, robust=True, check_std=True):
        x, y = np.asarray(self.position).T

        if check_std:
            rx, ry = check_slope(x, y)
        else:
            rx, ry = x, y

        if robust:
            slope, intercept = stats.siegelslopes(ry, rx)
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(rx, ry)

        if rx is x:
            return slope, -1, intercept
        else:
            return -1, slope, intercept

    @overload
    def align(self, points: NDArray, *, robust: bool = True, check_std: bool = True) -> np.ndarray: ...

    @overload
    def align(self, *, robust: bool = True, check_std: bool = True) -> AerialSurveyLine: ...

    def align(self, points=None, *, robust=True, check_std=True):
        if points is not None:
            position = np.asarray(points)
        else:
            position = self.position

        x = get_column(position, 0)
        y = get_column(position, 1)
        a, b, c = self.get_line_spec(robust=robust, check_std=check_std)
        if b == -1:
            y = a * x + c
        else:
            x = b * y + c

        if points is None:
            return AerialSurveyLine(cstack(x, y), self.data)
        else:
            return cstack(x, y)

    def __len__(self):
        return len(self.position)

    @overload
    def __getitem__(self, item: int) -> AerialSurveyPoint: ...

    @overload
    def __getitem__(self, item: slice | Collection[int]) -> AerialSurveyLine: ...

    def __getitem__(self, item):
        """对测线进行切片

        传入单个值索引时会返回测点对象

        默认按行号取值，如果需要取pandas的行索引，请使用 `pd.Index` 进行包装

        Parameters
        ----------
        item
            索引

        Returns
        -------
        sliced_line
            索引切片得到的新测线或测点
        """
        if self.data is not None:
            data = get_row(self.data, item)
        else:
            data = None

        position = get_row(self.position, item)

        if isinstance(item, int):
            return AerialSurveyPoint(position, data)
        else:
            return AerialSurveyLine(position, data)
