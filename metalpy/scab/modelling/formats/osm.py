from __future__ import annotations

from pathlib import Path
from xml import sax

import numpy as np

from metalpy.scab.modelling.shapes import Prism
from metalpy.scab.utils.proj import query_utm_transform_from_bounds
from metalpy.utils.type import not_none_or_default


class OSMHandler(sax.ContentHandler):
    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.ways = {}
        self.buildings = {}
        self.current = None
        self.current_way = None
        self.bounds = np.zeros(4, dtype=np.float64)

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


def load_osm(path: str | Path,
             level=0,
             default_height=10,
             height_map: dict[int, float] = None):
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

    Returns
    -------
    scene
        给定OSM文件定义的场景

    Notes
    -----
    OSM格式坐标默认为经纬度，通过pyproj库映射到UTM坐标系下，目前不支持跨UTM区
    """
    height_map = not_none_or_default(height_map, supplier=lambda: {})
    path = Path(path)

    parser = sax.make_parser()
    parser.setFeature(sax.handler.feature_namespaces, 0)
    handler = OSMHandler()
    parser.setContentHandler(handler)

    parser.parse(path)

    to_utm = query_utm_transform_from_bounds(handler.bounds)

    buildings = [
        Prism([to_utm.transform(*handler.nodes[ref]) for ref in b[:-1]],
              level, level + height_map.get(k, default_height))
        for k, b in handler.buildings.items()
    ]

    from metalpy.scab.modelling import Scene
    return Scene.of(*buildings), handler


class OSMFormat:
    @staticmethod
    def from_osm(
        path: str | Path,
        level=0,
        default_height=10,
        height_map: dict[int, float] = None,
        extras=False
    ):
        scene, extra = load_osm(path=path,
                                level=level,
                                default_height=default_height,
                                height_map=height_map)

        if extras:
            return scene, extra
        else:
            return scene
