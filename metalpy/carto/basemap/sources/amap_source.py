from __future__ import annotations

import warnings

from metalpy.carto.coords import wgs2gcj
from .simple_tile_map_source import SimpleTileMapSource


class AMapSource(SimpleTileMapSource):
    LangDefault = None
    LangChs = 'zh-CN'
    LangEn = 'en'
    LayerBackground = 1
    LayerRoads = 2
    LayerLabels = 4
    LayerAll = LayerBackground | LayerRoads | LayerLabels

    SiteRd = r'https://webrd0[1234].is.autonavi.com/appmaptile\?x=\{x\}&y=\{y\}&z=\{z\}'  # deprecated
    SiteSt = r'https://webst0[1234].is.autonavi.com/appmaptile\?x=\{x\}&y=\{y\}&z=\{z\}'  # deprecated
    SiteWprd = r'https://wprd0[1234].is.autonavi.com/appmaptile\?x=\{x\}&y=\{y\}&z=\{z\}'

    def __init__(self,
                 satellite=False,
                 layers: int | None = None,
                 transparent_background=False,
                 site=None,
                 lang=None,
                 scl=None,
                 style=None
                 ):
        """高德地图数据源

        Parameters
        ----------
        """
        if site is None:
            site = AMapSource.SiteWprd

        if lang is None:
            lang = AMapSource.LangChs

        if layers is None and not satellite:
            layers = AMapSource.LayerAll

        components = [site]
        if lang is not None:
            components.append(f'lang={lang}')

        if style is None:
            if satellite:
                if layers is not None and layers != 0:
                    warnings.warn('AMap source does not support'
                                  ' download satellite map with other layers.')
                style = 6
            elif transparent_background:
                style = 8
            else:
                style = 7

        components.append(f'style={style}')
        if layers is not None:
            components.append(f'ltype={layers}')

        if scl is not None:
            components.append(f'scl={scl}')

        super().__init__(
            '&'.join(components),
            google_map_style=True,
            regex=True
        )

    def warp_coordinates(self, coords):
        return wgs2gcj(*coords)
