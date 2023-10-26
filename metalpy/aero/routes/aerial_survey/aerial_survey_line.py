from __future__ import annotations

import numpy as np
from scipy import stats

from metalpy.utils.array_ops import get_column, cstack
from . import AerialSurvey
from metalpy.aero.utils.line_analysis import check_slope


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

    def align(self, robust=True, check_std=True):
        x = get_column(self.position, 0)
        y = get_column(self.position, 1)
        a, b, c = self.get_line_spec(robust=robust, check_std=check_std)
        if b == -1:
            y = a * x + c
        else:
            x = b * y + c

        return AerialSurveyLine(cstack(x, y), self.data)

    def __len__(self):
        return len(self.position)
