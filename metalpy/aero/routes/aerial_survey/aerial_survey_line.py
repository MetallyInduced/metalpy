from __future__ import annotations

import numpy as np
from scipy import stats

from . import AerialSurvey
from metalpy.aero.utils.line_analysis import check_slope


class AerialSurveyLine(AerialSurvey):
    def __init__(self, position, data):
        super().__init__(position, data)

    def get_line_spec(self):
        x, y = np.asarray(self.position).T
        rx, ry = check_slope(x, y)

        # slope, intercept, r_value, p_value, std_err = stats.linregress(rx, ry)
        slope, intercept = stats.siegelslopes(ry, rx)

        if rx is x:
            return slope, -1, intercept
        else:
            return -1, slope, intercept
