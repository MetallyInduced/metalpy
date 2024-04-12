from __future__ import annotations

import datetime
import json
import re
import warnings
from bisect import bisect, bisect_right, bisect_left
from collections import namedtuple, OrderedDict

import numpy as np
import requests

from metalpy.utils.bounds import Bounds
from metalpy.utils.file import file_cached
from .simple_tile_map_source import SimpleTileMapSource

# 版本年份，修订号
# 注：版本年份不一定和title中的发布日期一致，特例为2017r21对应2018-01-18。
Revision = namedtuple('Revision', ['year', 'rev'])
# 拍摄日期，影像来源，影像来源描述（拍摄卫星），分辨率，定位精度（与真实地理位置的偏差）
SourceMetaData = namedtuple('SourceMetaData', ['capture_date', 'source', 'desc', 'resolution', 'accuracy'])


class WaybackSource(SimpleTileMapSource):
    Sources: list[WaybackSource] = []
    SourcesByRnum: dict[int, WaybackSource] = {}

    Baseurl = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery'
    TileUrl = Baseurl + '/WMTS/1.0.0/default028mm/MapServer/tile/{rnum}/{level}/{row}/{col}'  # 默认瓦片地图信息地址
    TileInfoUrl = Baseurl + '/MapServer/tilemap/{rnum}/{level}/{row}/{col}'  # 默认瓦片地图信息地址

    MetaDataBaseurl = 'https://metadata.maptiles.arcgis.com/arcgis/rest/services'
    MetaDataUrl = MetaDataBaseurl + '/World_Imagery_Metadata_{year}_r{rev}/MapServer'  # 默认瓦片地图元数据地址

    # TODO: 读取官方WMTS服务
    #  https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/wmts/WMTSCapabilities.xml
    ConfigUrl = 'https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json'  # Wayback配置地址

    MAX_ZOOM = 23
    MIN_ZOOM = 10

    @staticmethod
    def get_latest():
        return WaybackSource.get_sources()[-1]

    def __init__(self, title, date, rev: Revision, rnum=None, tile_url=None, metadata_url=None):
        """Wayback (esri) 地图数据源

        References
        ----------
        World Imagery Wayback: https://livingatlas.arcgis.com/wayback/
        """
        self.title = title
        self.date = date
        self.revision = rev
        self.release_num = rnum

        if tile_url is None:
            tile_url = WaybackSource.format_tile_url(rnum)

        if metadata_url is None:
            metadata_url = WaybackSource.format_metadata_url(self.revision)

        self.tile_url = tile_url
        self.metadata_url = metadata_url

        super().__init__(
            tile_url
        )

    @property
    def year(self):
        return self.revision.year

    @property
    def rev(self):
        return self.revision.rev

    def query_metadata(self, mercator_bounds: Bounds, level, session=None):
        """查询区域内卫星地图的元数据，包括影像采集时间，影像源，采集卫星，分辨率，精度等。

        Parameters
        ----------
        mercator_bounds
            区域边界
        level
            瓦片地图等级
        session
            requests会话，若为空则直接使用request

        Returns
        -------
        metas
            元数据列表，因为区域内可能包含多个来源的数据
        """
        mercator_bounds = Bounds(mercator_bounds)
        geometry = {
            'spatialReference': {'latestWkid': 3857, 'wkid': 102100},
            'xmin': mercator_bounds.xmin,
            'xmax': mercator_bounds.xmax,
            'ymin': mercator_bounds.ymin,
            'ymax': mercator_bounds.ymax,
        }
        params = {
            'f': 'json',
            'outFields': ','.join(['SRC_DATE2', 'NICE_DESC', 'SRC_DESC', 'SRC_RES', 'SRC_ACC']),
            'geometry': json.dumps(geometry),
            'geometryType': 'esriGeometryEnvelope',
            'spatialRel': 'esriSpatialRelIntersects',
            'returnGeometry': 'false'
        }
        layer_id = WaybackSource.to_metadata_layer_id(level)

        if session is None:
            session = requests

        resp = session.get(self.metadata_url + f'/{layer_id}/query', params=params)
        metas = resp.json()

        metas = [WaybackSource.to_metadata(meta) for meta in metas['features']]

        return metas

    @staticmethod
    def query_tile_changes(mercator_bounds, level, session=None, region_samples: int | float | None = 0.2):
        """查询区域内瓦片存在变更的时间点（仍可能存在重复）

        Parameters
        ----------
        mercator_bounds
            区域边界
        level
            瓦片地图等级
        session
            requests会话，若为空则内部创建
        region_samples
            区域采样，
            如果为整数则为固定个数采样，
            如果为小数则按比例采样，
            如果为空则不进行采样

        Returns
        -------
        wayback_sources
            区域内瓦片发生变更的时间点的地图源

        TODO: 实现并行查询
        TODO: 实现进一步去重
          （https://github.com/vannizhang/wayback/blob/master/src/core/WaybackManager/ChangeDetector.ts#L221）
        """
        from metalpy.carto.basemap.constants import WebMercator
        _1day = datetime.timedelta(days=1)
        sources = OrderedDict()
        latest_rnum = WaybackSource.get_sources()[-1].release_num

        if session is None:
            session = requests.session()

        tiles = list(WebMercator.iter_tiles(mercator_bounds, level))
        if isinstance(region_samples, float):
            assert 0 < region_samples < 1, 'Proportional region samples must be within (0.0, 1.0).'
            region_samples = int(len(tiles) * region_samples)

        if isinstance(region_samples, int):
            assert region_samples > 1, 'Fixed region samples must be larger than 0.'
            if region_samples < len(tiles):
                sample_indices = np.linspace(0, len(tiles) - 1, region_samples).astype(int)
                tiles = [tiles[i] for i in sample_indices]

        for col, row in tiles:
            last_release_num = latest_rnum

            while last_release_num is not None:
                url = WaybackSource.format_tile_info_url(last_release_num, level, col, row)
                info = session.get(url).json()
                if 'select' in info and len(info['select']) > 0:
                    last_release_num = info['select'][0]
                    release = sources[last_release_num] = WaybackSource.find_by_release_num(last_release_num)
                else:
                    release = WaybackSource.find_by_release_num(last_release_num)

                if 'data' in info and info['data'][0]:
                    last_release = WaybackSource.find_latest_before(release.date - _1day)
                    if last_release is not None:
                        last_release_num = last_release.release_num
                    else:
                        last_release_num = None
                else:
                    last_release_num = None

        ret = list(sources.values())
        ret.sort()

        return ret

    @staticmethod
    def find_by_rev(year: int | None = None, rev: int | Revision = None):
        if year is None:
            year = rev.year
            rev = rev.rev
        if not isinstance(rev, Revision):
            rev = Revision(year=year, rev=rev)
        return WaybackSource._find_by(rev)

    @staticmethod
    def find_by_date(date):
        return WaybackSource._find_by(date)

    @staticmethod
    def find_by_release_num(rnum):
        return WaybackSource.get_sources_from_rnum().get(rnum)

    @staticmethod
    def findall_by_year(year):
        sources = WaybackSource.get_sources()
        left = bisect(sources, datetime.date(year, 1, 1))
        right = bisect(sources, datetime.date(year + 1, 1, 1), lo=left)

        return sources[left:right]

    @staticmethod
    def findall_before(date: datetime.date):
        sources = WaybackSource.get_sources()
        right = bisect_left(sources, date)
        return sources[:right]

    @staticmethod
    def findall_after(date: datetime.date):
        sources = WaybackSource.get_sources()
        left = bisect_right(sources, date)
        return sources[left:]

    @staticmethod
    def find_latest_before(date: datetime.date):
        sources = WaybackSource.get_sources()
        right = bisect_right(sources, date)
        if right <= 0:
            return None
        return sources[right - 1]

    @staticmethod
    def find_earliest_after(date: datetime.date):
        sources = WaybackSource.get_sources()
        left = bisect_left(sources, date)
        if left > len(sources):
            return None
        return sources[left]

    @staticmethod
    def _find_by(date_or_rev):
        sources = WaybackSource.get_sources()
        index = bisect_left(sources, date_or_rev)
        if sources[index] != date_or_rev:
            return None
        return sources[index]

    @staticmethod
    def from_wayback_item(rnum, item_info):
        match = re.compile(r'(\d+)_r(\d+)').search(item_info['metadataLayerUrl'])
        revision = Revision(year=int(match[1]), rev=int(match[2]))

        date = re.compile(r'\d{4}-\d{2}-\d{2}').search(item_info['itemTitle'])[0]
        date = datetime.date.fromisoformat(date)

        urls = {}
        tile_url = item_info['itemURL']
        if tile_url != WaybackSource.format_tile_url(rnum):
            warnings.warn('Tile url updated!'
                          ' Please open an issue on `https://github.com/yanang007/metalpy/issues`'
                          ' to report this change.')
            urls['tile_url'] = tile_url

        metadata_url = item_info['metadataLayerUrl']
        if metadata_url != WaybackSource.format_metadata_url(revision):
            warnings.warn('Metadata url updated!'
                          ' Please open an issue on `https://github.com/yanang007/metalpy/issues`'
                          ' to report this change.')
            urls['metadata_url'] = metadata_url

        return WaybackSource(
            title=item_info['itemTitle'],
            date=date,
            rev=revision,
            rnum=int(rnum),
            **urls
        )

    @staticmethod
    def get_sources(force_update=False):
        if len(WaybackSource.Sources) == 0:
            WaybackSource.Sources = []
            configs = WaybackSource.get_config.prepare(update=force_update)()
            for rnum, info in configs.items():
                source = WaybackSource.from_wayback_item(rnum, info)
                WaybackSource.Sources.append(source)
                WaybackSource.SourcesByRnum[source.release_num] = source

        WaybackSource.Sources.sort()
        return WaybackSource.Sources

    @staticmethod
    def get_sources_from_rnum(force_update=False):
        WaybackSource.get_sources(force_update)
        return WaybackSource.SourcesByRnum

    @staticmethod
    @file_cached(ttl=datetime.timedelta(days=3))
    def get_config():
        resp = requests.get(WaybackSource.ConfigUrl)
        return resp.json()

    @staticmethod
    def format_tile_url(rnum):
        return WaybackSource.TileUrl.replace('{rnum}', str(rnum))

    @staticmethod
    def format_tile_info_url(rnum, level, col, row):
        return WaybackSource.TileInfoUrl.format(rnum=rnum, level=level, col=col, row=row)

    @staticmethod
    def format_metadata_url(rev: Revision):
        return WaybackSource.MetaDataUrl.format(year=rev.year, rev=f'{rev.rev:02}')

    @staticmethod
    def to_metadata_layer_id(level):
        layer_id = WaybackSource.MAX_ZOOM - level
        layer_id = max(layer_id, 0)
        layer_id = min(layer_id, WaybackSource.MAX_ZOOM - WaybackSource.MIN_ZOOM)

        return layer_id

    @staticmethod
    def to_metadata(meta):
        meta = meta['attributes']
        date = datetime.datetime.fromtimestamp(int(meta['SRC_DATE2']) // 1000)
        date = date.astimezone(datetime.timezone.utc)
        return SourceMetaData(
            capture_date=date,
            source=meta['NICE_DESC'],
            desc=meta['SRC_DESC'],
            resolution=meta['SRC_RES'],
            accuracy=meta['SRC_ACC'],
        )

    def __eq__(self, other):
        return self._cmp(other) == 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def __ge__(self, other):
        return self._cmp(other) >= 0

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def __ne__(self, other):
        return self._cmp(other) != 0

    def _cmp(self, other):
        if isinstance(other, WaybackSource):
            return (self.date - other.date).days
        elif isinstance(other, datetime.date):
            return (self.date - other).days
        elif isinstance(other, Revision):
            dyear = self.year - other.year
            if dyear == 0:
                return self.rev - other.rev
            else:
                return dyear
        else:
            return self.date - other
