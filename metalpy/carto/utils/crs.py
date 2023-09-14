from __future__ import annotations

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pyproj import CRS


CRSLike = Union['CRS', str]


def check_crs(crs: CRSLike):
    from pyproj import CRS

    if isinstance(crs, str):
        crs = CRS.from_string(crs)

    return crs
