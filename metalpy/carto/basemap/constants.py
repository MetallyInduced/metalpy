from functools import lru_cache

import numpy as np
from pyproj import CRS, Transformer

from metalpy.utils.bounds import Bounds


class WebMercator:
    """适用于四叉树组织的WebMercator坐标系相关计算

    TODO：实现二分-四叉树组织相关计算
    """
    DirectionAll = 'all'
    WGS84 = CRS.from_string('WGS 84')
    WebMercator = CRS.from_string('WGS 84 / Pseudo-Mercator')

    Perimeter = 2 * np.pi * WebMercator.ellipsoid.semi_major_metre
    WGS84_WebMercator = Transformer.from_crs(WGS84, WebMercator, always_xy=True)

    @staticmethod
    def wgs84_to_pseudo_mercator(long, lat):
        """从WGS84转换为WebMercator坐标系，
        相比于EPSG:3857的Pseudo-Mercator，
        WebMercator坐标系通过偏移保证坐标为正
        """
        x, y = WebMercator.WGS84_WebMercator.transform(long, lat)

        return x, y

    @staticmethod
    @lru_cache
    def tiles_in_axis(direction, level):
        """计算指定方向上在level等级下的tile数

        Parameters
        ----------
        direction
            方向，0代表x东西，1代表y南北
        level
            tile等级

        Returns
        -------
        ret
            在level等级下给定方向上最大tile数
        """
        return 2 ** level

    @staticmethod
    def tile_width(direction, level):
        return WebMercator.Perimeter / WebMercator.tiles_in_axis(direction, level)

    @staticmethod
    def pseudo_mercator_to_tile(coord, level):
        tile_width = WebMercator.tile_width(WebMercator.DirectionAll, level)
        return np.floor((coord + WebMercator.Perimeter / 2) / tile_width).astype(np.int32)

    @staticmethod
    def pseudo_mercator_bounds_to_tile(bounds, level):
        tile_width = WebMercator.tile_width(WebMercator.DirectionAll, level)
        return np.floor(Bounds(bounds + WebMercator.Perimeter / 2) / tile_width).astype(np.int32)

    @staticmethod
    def tile_bounds_to_pseudo_mercator(bounds, level):
        """将tile坐标边界转换为WebMercator坐标边界

        Parameters
        ----------
        bounds
            tile坐标边界
        level
            地图等级

        Returns
        -------
        web_mercator_bounds
            WebMercator坐标边界

        Notes
        -----
        由于tile坐标指向的为一个瓦片而非点，因此边界转换时会取最小位置瓦片的下边界和最大位置瓦片的上边界
        """
        tile_width = WebMercator.tile_width(WebMercator.DirectionAll, level)
        return (Bounds(bounds) + [0, 1, 0, 1]) * tile_width - WebMercator.Perimeter / 2

    @staticmethod
    def iter_tiles(mercator_bounds, level):
        mercator_bounds = np.array(mercator_bounds)
        tile_bounds = WebMercator.pseudo_mercator_bounds_to_tile(mercator_bounds, level)

        for x in range(tile_bounds[0], tile_bounds[1] + 1):
            for y in range(tile_bounds[2], tile_bounds[3] + 1):
                yield x, y

    @staticmethod
    def warp_tile_coord(x, y, z, bottom_left_as_origin=False):
        """WMTS规定北纬85.05°为零点，而carto中约定南纬85.05°为零点，
        因此Y方向tile坐标会有区别

        Parameters
        ----------
        x,y
            左下原点标准下x、y方向tile坐标
        z
            tile等级
        bottom_left_as_origin
            是否采用左下角作为tile原点

        Returns
        -------
        tile_coord_in_top_left_as_origin
            给定约定下的tile坐标
        """
        if not bottom_left_as_origin:
            y = WebMercator.tiles_in_axis(1, z) - y - 1
        return x, y
