from __future__ import annotations

from typing import Union

from pyproj import CRS


CRSLike = Union[CRS, str]


def check_crs(crs: CRSLike):
    if isinstance(crs, str):
        crs = CRS.from_string(crs)

    return crs
