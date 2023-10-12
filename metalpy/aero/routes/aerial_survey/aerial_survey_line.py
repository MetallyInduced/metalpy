from __future__ import annotations

import numpy as np
from scipy import stats

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

    def __len__(self):
        return len(self.position)
