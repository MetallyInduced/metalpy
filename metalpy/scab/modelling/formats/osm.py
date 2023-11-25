from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, overload, Literal, Sequence
from xml import sax

import numpy as np
import pandas as pd

from metalpy.carto.coords import Coordinates
from metalpy.carto.coords.coordinates import CRSQuery
from metalpy.scab.modelling.shapes import Prism
from metalpy.scab.utils.proj import query_utm_crs_from_bound
from metalpy.utils.bounds import Bounds
from metalpy.utils.file import PathLike, make_cache_file_path, PathLikeType
from metalpy.utils.type import not_none_or_default

if TYPE_CHECKING:
    from metalpy.scab.modelling import Scene


class OSMHandler(sax.ContentHandler):
    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.ways = {}
        self.buildings = {}
        self.current = None
        self.current_way = None
        self.bounds = np.zeros(4, dtype=np.float64)

    @property
    def utm_crs(self):
        return query_utm_crs_from_bound(self.bounds)

    def startElement(self, tag, attributes):
        if tag == 'bounds':
            self.bounds[0] = float(attributes['minlon'])
            self.bounds[1] = float(attributes['maxlon'])
            self.bounds[2] = float(attributes['minlat'])
            self.bounds[3] = float(attributes['maxlat'])
        elif tag == 'node':
            node_id = int(attributes['id'])
            self.nodes[node_id] = (float(attributes['lon']), float(attributes['lat']))
            self.current = tag
        elif tag == 'way':
            self.current_way = int(attributes['id'])
            self.current = tag
        elif tag == 'nd':
            if self.current == 'way':
                self.ways.setdefault(self.current_way, []).append(int(attributes['ref']))
        elif tag == 'tag':
            if self.current == 'way':
                if attributes['k'] == 'building':
                    self.buildings[self.current_way] = self.ways[self.current_way]

    def endElement(self, tag):
        if tag in ("node", 'way'):
            self.current = None
            self.current_way = None


def load_osm(path: PathLike,
             level=0,
             default_height=10,
             height_map: dict[int, float] = None,
             dest_crs=Coordinates.SearchUTM
             ) -> tuple[list[Prism], OSMHandler]:
    """从OSM格式xml文件生成Scene实例

    Parameters
    ----------
    path
        OSM文件路径
    level
        目标区域的基准面海拔
    default_height
        OSM文件未指定高度的建筑物使用该默认高度替换
    height_map
        通过建筑物id，为建筑物指定高度
    dest_crs
        转换的目标坐标系（OSM文件中的建筑节点为WGS 84坐标系）

    Returns
    -------
    (scene, handler)
        给定OSM文件定义的建筑物列表和OSM文件相关信息

    Notes
    -----
    OSM格式坐标默认为经纬度，默认映射到UTM坐标系下，目前不支持跨UTM区
    """
    height_map = not_none_or_default(height_map, supplier=lambda: {})
    path = Path(path)

    parser = sax.make_parser()
    parser.setFeature(sax.handler.feature_namespaces, 0)
    handler = OSMHandler()
    parser.setContentHandler(handler)

    parser.parse(path)

    nodes = pd.DataFrame(handler.nodes).T
    Coordinates(nodes).warp(query=dest_crs, inplace=True)

    buildings = [
        Prism(nodes.loc[b[:-1]].to_numpy(),
              level, level + height_map.get(k, default_height))
        for k, b in handler.buildings.items()
    ]

    return buildings, handler


class OSMFormat:
    @staticmethod
    @overload
    def from_osm(
            path_or_bounds: PathLike | None = None,
            *,
            path: PathLike | None = None,
            level=0,
            default_height=10,
            height_map: dict[int, float] = None,
            extras: Literal[True],
            dest_crs: CRSQuery = Coordinates.SearchUTM
    ) -> tuple[Scene, OSMHandler]:
        ...

    @staticmethod
    @overload
    def from_osm(
            path_or_bounds: PathLike | None = None,
            *,
            path: PathLike | None = None,
            level=0,
            default_height=10,
            height_map: dict[int, float] = None,
            extras: Literal[False] = False,
            dest_crs: CRSQuery = Coordinates.SearchUTM
    ) -> Scene:
        ...

    @staticmethod
    @overload
    def from_osm(
            path_or_bounds: PathLike | None = None,
            *,
            path: PathLike | None = None,
            level=0,
            default_height=10,
            height_map: dict[int, float] = None,
            extras: bool,
            dest_crs: CRSQuery = Coordinates.SearchUTM
    ) -> tuple[Scene, OSMHandler] | Scene:
        ...

    @staticmethod
    def from_osm(
            path_or_bounds: PathLike | Sequence | None = None,
            *,
            path: PathLike | None = None,
            level=0,
            default_height=10,
            height_map: dict[int, float] = None,
            extras=False,
            dest_crs: CRSQuery = Coordinates.SearchUTM
    ) -> tuple[Scene, OSMHandler] | Scene:
        """从OSM格式xml文件生成Scene实例

        Parameters
        ----------
        path_or_bounds
            需要载入的OSM文件路径。
            或需要下载的建筑物区域WGS 84边界，
            格式依次为[minLong, maxLong, minLat, maxLat]
        path
            OSM文件缓存路径，只有
        level
            目标区域的基准面海拔
        default_height
            OSM文件未指定高度的建筑物使用该默认高度替换
        height_map
            通过建筑物id，为建筑物指定高度
        extras
            指示是否返回OSM解析中间信息
        dest_crs
            转换的目标坐标系（OSM文件中的建筑节点为WGS 84坐标系）

        Returns
        -------
        (scene, handler)
            给定OSM文件定义的建筑物列表和OSM文件相关信息

        Notes
        -----
        OSM格式坐标默认为经纬度，默认映射到UTM坐标系下，目前不支持跨UTM区
        """
        if isinstance(path_or_bounds, PathLikeType):
            path = Path(path)
        else:
            bounds = path_or_bounds

            bounds = Bounds(np.asarray(bounds)[:4])
            b_str = ','.join([str(i) for i in bounds.as_corners().ravel()])

            if path is None:
                path = make_cache_file_path(f'map{bounds}.osm')
            else:
                path = Path(path)

            if not path.exists():
                import requests
                url = f'https://overpass-api.de/api/map?bbox={b_str}'
                data = requests.get(url).content.decode('utf-8')
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(data)

        assert path is not None, (
            'Either'
            ' a path to a local osm file'
            ' or bounds of a region to download osm data'
            ' must be specified.'
        )

        buildings, extra = load_osm(
            path=path,
            level=level,
            default_height=default_height,
            height_map=height_map,
            dest_crs=dest_crs
        )

        from metalpy.scab.modelling import Scene
        scene = Scene.of(*buildings)

        if extras:
            return scene, extra
        else:
            return scene
