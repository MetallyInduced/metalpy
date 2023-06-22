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
    PseudoMercator = CRS.from_string('WGS 84 / Pseudo-Mercator')

    Perimeter = 2 * np.pi * PseudoMercator.ellipsoid.semi_major_metre
    WGS84_PseudoMercator = Transformer.from_crs(WGS84, PseudoMercator, always_xy=True)
    PseudoMercator_WGS84 = Transformer.from_crs(PseudoMercator, WGS84, always_xy=True)

    @staticmethod
    def wgs84_to_pseudo_mercator(long, lat):
        """从WGS84转换为Pseudo-Mercator坐标系
        """
        x, y = WebMercator.WGS84_PseudoMercator.transform(long, lat)

        return x, y

    @staticmethod
    def pseudo_mercator_to_wgs84(x, y):
        """从Pseudo-Mercator转换为WGS84坐标系
        """
        long, lat = WebMercator.PseudoMercator_WGS84.transform(x, y)

        return long, lat

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
    def pseudo_mercator_bounds_to_tile(bounds, level):
        """将Pseudo-Mercator坐标边界转换为Tile序号边界。

        Parameters
        ----------
        bounds
            Pseudo-Mercator坐标边界
        level
            缩放等级

        Returns
        -------
        tile_bounds
            Tile序号边界

        Notes
        -----
        由于原点方向不一致，转换后y方向上下边界会交换，例如TileYMax对应MercatorYMin。
        """
        tile_width = WebMercator.tile_width(WebMercator.DirectionAll, level)
        y_tiles = WebMercator.tiles_in_axis(WebMercator.DirectionAll, level)

        ret = np.floor(Bounds(bounds + WebMercator.Perimeter / 2) / tile_width).astype(np.int32)
        ret.ymin, ret.ymax = y_tiles - ret.ymax - 1, y_tiles - ret.ymin - 1

        return ret

    @staticmethod
    def tile_bounds_to_pseudo_mercator(bounds, level):
        """将Tile序号边界转换为Pseudo-Mercator坐标边界

        Parameters
        ----------
        bounds
            Tile序号边界
        level
            缩放等级

        Returns
        -------
        pseudo_mercator_bounds
            Pseudo-Mercator坐标边界

        Notes
        -----
        由于tile坐标指向的为一个瓦片而非点，因此边界转换时会取最小位置瓦片的下边界和最大位置瓦片的上边界

        由于原点方向不一致，转换后y方向上下边界会交换，例如MercatorYMin对应TileYMax。
        """
        tile_width = WebMercator.tile_width(WebMercator.DirectionAll, level)
        y_tiles = WebMercator.tiles_in_axis(WebMercator.DirectionAll, level)

        bounds = Bounds(bounds).copy()
        bounds.ymin, bounds.ymax = y_tiles - bounds.ymax - 1, y_tiles - bounds.ymin - 1

        return (bounds + [0, 1, 0, 1]) * tile_width - WebMercator.Perimeter / 2

    @staticmethod
    def to_bounding_tiles_mercator_bounds(mercator_bounds, level):
        """将Pseudo-Mercator边界坐标扩大为所涉及的Tile的Pseudo-Mercator边界
        
        Parameters
        ----------
        mercator_bounds
            需要的地图边界
        level
            缩放等级

        Returns
        -------
        tiles_mercator_bounds
            所涉及Tile的Pseudo-Mercator边界
        """
        tile_bounds = WebMercator.pseudo_mercator_bounds_to_tile(mercator_bounds, level)
        tiles_mercator_bounds = WebMercator.tile_bounds_to_pseudo_mercator(tile_bounds, level)

        return tiles_mercator_bounds

    @staticmethod
    def iter_tiles(mercator_bounds, level):
        mercator_bounds = np.array(mercator_bounds)
        tile_bounds = WebMercator.pseudo_mercator_bounds_to_tile(mercator_bounds, level)

        for x in range(tile_bounds[0], tile_bounds[1] + 1):
            for y in range(tile_bounds[2], tile_bounds[3] + 1):
                yield x, y

    @staticmethod
    def warp_tile_coord(x, y, z, bottom_left_as_origin=False):
        """WMTS规定北纬85.05°为零点，该函数用于根据给定参数，转换为对应的南纬85.05°为零点或不转换

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
        if bottom_left_as_origin:
            y = WebMercator.tiles_in_axis(1, z) - y - 1
        return x, y
