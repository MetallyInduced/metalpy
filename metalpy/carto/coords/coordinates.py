from __future__ import annotations

import re
import warnings
from typing import Union

import numpy as np
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info, query_crs_info

from metalpy.carto.utils.crs import check_crs, CRSLike


CRSQuery = Union[str, re.Pattern, callable]


class Coordinates(np.ndarray):
    """用于表示一个带有坐标系信息的坐标，支持进行便捷坐标转换

    Examples
    --------
    >>> coord = Coordinates([[139.6817, 35.6794], [139.6917, 35.6894]], crs='WGS 84')
    >>> coord.warp(query=Coordinates.SearchUTM)
    Coordinates([[ 380702.31913457, 3949190.73587477],
                [ 381622.08225664, 3950287.81606255]], crs='WGS 84 / UTM zone 54N')
    """
    WGS_84 = CRS.from_string('WGS 84')
    SearchUTM = '__SearchUTM'

    def __new__(cls, arr, crs: CRSLike = WGS_84):
        return np.asarray(arr).view(Coordinates).with_crs(crs)

    def __array_finalize__(self, obj, **kwargs):
        self._init(getattr(obj, 'crs', None))

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        inputs = tuple(inp.to_numpy() if isinstance(inp, Coordinates) else inp for inp in inputs)
        if out is not None:
            out = tuple(o.to_numpy() if isinstance(o, Coordinates) else o for o in out)
        ret = super().__array_ufunc__(ufunc, method, *inputs, out=out, **kwargs)

        if ret is NotImplemented:
            return NotImplemented

        if ret.ndim == self.ndim:
            return Coordinates(ret, crs=self.crs)
        else:
            return ret

    def __getitem__(self, index):
        return self.to_numpy()[index]

    def __setitem__(self, index, val):
        self.to_numpy()[index] = val

    def __repr__(self):
        ret = super().__repr__()
        return ret.replace(')', f', crs={repr(self.crs.name)})')

    def _init(self, crs=None):
        if crs is None:
            crs = 'WGS 84'

        self.crs = crs
        assert self.ndim in (1, 2) and self.shape[-1] in (2, 3), 'Coordinates supports only 2D or 3D points array.'

    @property
    def crs(self) -> CRS:
        return getattr(self, '_crs', None)

    @crs.setter
    def crs(self, crs_or_name: CRSLike):
        setattr(self, '_crs', check_crs(crs_or_name))

    def to_numpy(self):
        return self.view(np.ndarray)

    @property
    def single(self):
        return self.ndim == 1

    @property
    def x(self):
        if self.single:
            return self[0]
        else:
            return self[:, 0]

    @x.setter
    def x(self, val):
        if self.single:
            self[0] = val
        else:
            self[:, 0] = val

    @property
    def y(self):
        if self.single:
            return self[1]
        else:
            return self[:, 1]

    @y.setter
    def y(self, val):
        if self.single:
            self[1] = val
        else:
            self[:, 1] = val

    @property
    def z(self):
        if self.single:
            return self[2]
        else:
            return self[:, 2]

    @z.setter
    def z(self, val):
        if self.single:
            self[2] = val
        else:
            self[:, 2] = val

    @staticmethod
    def warn_invalid_modification():
        warnings.warn('Modifying dims of Coordinates is not allowed.')

    def reshape(self, *_, **__):
        Coordinates.warn_invalid_modification()

    def with_crs(self, crs):
        self.crs = crs
        return self

    def warp(self,
             crs: CRSLike | None = None,
             query: CRSQuery | None = None,
             inplace=False):
        """将所有坐标转换到指定的另一坐标系下

        Parameters
        ----------
        crs
            新的坐标系
        query
            按指定模式搜索坐标系
        inplace
            指定转换是否就地操作

        Returns
        -------
        ret
            新坐标系下的坐标集合

        Notes
        -----
        query支持以下方式进行搜索:

        * 指定Coordinates.SearchUTM以搜索UTM分区

        * 指定字符串`CGCS2000 / 3-degree Gauss-Kruger zone`以包含指定串的坐标系

        * 指定正则模式`re.compile('CGCS2000.*3-degree.*Gauss-Kruger.*')`以搜索匹配的坐标系

        * 指定函数`lambda crs: crs.code == 3857`以搜索匹配的坐标系
        """
        if isinstance(query, CRS):
            crs = query
            query = None

        if crs is None:
            assert query is not None, 'Either `crs` or `query` must be specified.'

            if self.single:
                bounds = np.asarray(self).repeat(2)
            else:
                bounds = np.c_[self.min(axis=0), self.max(axis=0)].ravel()
            bounds2d = bounds[:4]

            if self.crs == Coordinates.WGS_84:
                wgs84_bounds = bounds2d
            else:
                to_wgs84 = Transformer.from_crs(self.crs, Coordinates.WGS_84, always_xy=True)
                wgs84_bounds = np.c_[
                    to_wgs84.transform(*bounds2d[::2]),
                    to_wgs84.transform(*bounds2d[1::2])
                ].ravel()

            l, r, b, t = wgs84_bounds
            area_of_interest = AreaOfInterest(
                west_lon_degree=l,
                east_lon_degree=r,
                south_lat_degree=b,
                north_lat_degree=t
            )

            if query is Coordinates.SearchUTM:
                crs_list = query_utm_crs_info(
                    datum_name='WGS 84',
                    area_of_interest=area_of_interest,
                )
            else:
                crs_list = query_crs_info(area_of_interest=area_of_interest)
                if isinstance(query, re.Pattern):
                    crs_list = [crs for crs in crs_list if query.match(crs.name)]
                elif isinstance(query, str):
                    # e.g. CGCS2000 / 3-degree Gauss-Kruger zone
                    crs_list = [crs for crs in crs_list if query in crs.name]
                elif callable(query):
                    crs_list = [crs for crs in crs_list if query(crs)]
                else:
                    crs_list = [crs for crs in crs_list if query == crs]

            assert len(crs_list) == 1, \
                f'Multiple CRS-s found by `{query}`:\n' + Coordinates.format_crs_list_str(crs_list)

            crs = CRS.from_epsg(crs_list[0].code)
        else:
            crs = check_crs(crs)

        transform = Transformer.from_crs(self.crs, crs, always_xy=True)

        if inplace:
            target = self
        else:
            target = self.copy()

        target.x, target.y = transform.transform(target.x, target.y)

        return target

    @staticmethod
    def format_crs_list_str(crs_list):
        return '\n'.join(['    - ' + crs.name for crs in crs_list])
