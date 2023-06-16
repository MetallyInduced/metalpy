from __future__ import annotations

import contextlib
import os
import warnings
from io import BytesIO
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Lock
from typing import Iterable

import numpy as np
import requests
import tqdm
from PIL import Image

from metalpy.utils.file import make_cache_directory_path, ensure_filepath, PathLike
from metalpy.utils.bounds import Bounds
from metalpy.utils.dhash import dhash
from metalpy.utils.path import pathencode

from .constants import WebMercator
from .geo_image import GeoImage
from .geo_image_ref_system import GeoImageRefSystem
from .sources.tile_map_source import TileMapSource
from .sources.tile_locator import TileLocator


class MapDownloader:
    """
    References
    ----------
    https://github.com/cutecore/google_map_satellite_download
    https://github.com/Icemap/go-map-downloader
    """
    def __init__(self, map_source: TileMapSource, n_threads=50, max_retries=3):
        # WMTS only for now
        self.map_source = map_source
        self.pool = ThreadPool(n_threads)

        self.max_retries = max_retries

        self.tile_locators: list[TileLocator] | None = None
        self.running = False
        self.session: requests.Session | None = None
        self.canvas_lock = Lock()
        self.progress = None

    def download(self, west_lon=None, east_lon=None, south_lat=None, north_lat=None,
                 bounds=None, levels=(18,),
                 combine: bool | PathLike | None = None,
                 crop: bool | PathLike | None = True,
                 geotiff: PathLike | None = None,
                 cache: bool | PathLike | None = None) -> GeoImage:
        """

        Parameters
        ----------
        west_lon, east_lon, south_lat, north_lat
            下载地图的西、东、南、北边界WGS 84坐标
        bounds
            下载地图的边界WGS 84坐标，以[lon_min, lon_max, lat_min, lat_max]顺序提供
        levels
            需要下载的地图等级
        combine
            下载的全部关联瓦片拼接后的存储路径
        crop
            下载的全部关联瓦片拼接后按边界裁剪结果的存储路径
        geotiff
            瓦片结果（是否裁剪后取决于是否crop）绑定地理坐标后的存储路径
        cache
            瓦片及中间结果的缓存路径

        Returns
        -------
        map_data
            处理后的地图数据

        Notes
        -----
        y方向有多次坐标转换，原因为WMTS标准的原点在左下，而图像的原点在左上，拼接以及裁剪时需要将y方向坐标镜像翻转
        """
        ret = None
        saved_path = None

        if bounds is not None:
            bounds = Bounds(bounds)
        else:
            bounds = Bounds([west_lon, east_lon, south_lat, north_lat])

        bounds.origin = self.map_source.warp_coordinates(bounds.origin)
        bounds.end = self.map_source.warp_coordinates(bounds.end)

        if not isinstance(levels, Iterable):
            levels = (levels,)

        mercator_bounds = WebMercator.wgs84_to_pseudo_mercator(bounds[0:2], bounds[2:4])
        mercator_bounds = np.concatenate(mercator_bounds)

        with self.activate_pool():
            for level in levels:
                tile_bounds = WebMercator.pseudo_mercator_bounds_to_tile(mercator_bounds, level)
                tiles_mercator_bounds = WebMercator.tile_bounds_to_pseudo_mercator(tile_bounds, level)
                ref_system = None
                for tile in self.download_geo_tiles(mercator_bounds, level, cache):
                    if ref_system is None:
                        ref_system = tile.ref_system
                        if combine or (geotiff and not crop):
                            canvas = ref_system.create_image(
                                geo_bounds=tiles_mercator_bounds
                            )
                        elif crop:
                            canvas = ref_system.create_image(
                                geo_bounds=mercator_bounds
                            )
                    canvas.paste(tile)

                ret = canvas

                if combine:
                    combined = canvas
                    if not isinstance(combine, bool):
                        saved_path = MapDownloader.encode_filename_with_level(
                            combine, level, levels, 'combined'
                        )
                        combined.save(saved_path)

                if crop:
                    if combine:
                        cropped = combined.crop(geo_bounds=mercator_bounds)
                    else:
                        cropped = canvas

                    ret = cropped

                    if not isinstance(crop, bool):
                        saved_path = MapDownloader.encode_filename_with_level(
                            crop, level, levels, 'cropped'
                        )
                        cropped.save(saved_path)

                if geotiff:
                    geotiff_save_path = MapDownloader.encode_filename_with_level(
                        geotiff, level, levels, 'geotiff'
                    )

                    ret.save_geo_tiff(geotiff_save_path)
                    # if saved_path is None:
                    #     ret.save_geo_tiff(geotiff_save_path)
                    # else:
                    #     ret.apply_geo_info(saved_path, geotiff_save_path)

        return ret

    def download_geo_tiles(self,
                           mercator_bounds,
                           level,
                           cache: bool | PathLike | None = None,
                           ref_system=None):
        tiles = list(WebMercator.iter_tiles(mercator_bounds, level))
        force_cache, cache_path = self.get_cache_path(cache)

        _cache_path = None
        if force_cache or len(tiles) > 100:
            _cache_path = cache_path

        tile_bounds = WebMercator.pseudo_mercator_bounds_to_tile(mercator_bounds, level)
        tiles_mercator_bounds = WebMercator.tile_bounds_to_pseudo_mercator(tile_bounds, level)

        tile_extent = tiles_mercator_bounds.extent
        tile_count = tile_bounds.extent + 1
        col0, row0 = tile_bounds.xmin, tile_bounds.ymin

        tile_size = None

        for col, row, tile in self.download_tiles(tiles, level, _cache_path):
            if tile_size is None:
                tile_size = tile.size

            if ref_system is None:
                combined_image_size = (
                    tile_size[0] * tile_count[0],
                    tile_size[1] * tile_count[1]
                )
                ref_system = GeoImageRefSystem.of_unit_edge_bounds(
                    tiles_mercator_bounds.origin,
                    tile_extent / combined_image_size,
                    crs=WebMercator.WebMercator
                )

            geotile = ref_system.map_image(tile, offset=(
                (col - col0) * tile_size[0],
                (row - row0) * tile_size[1]
            ))

            yield geotile

    def get_cache_path(self, cache):
        if cache is not None and not isinstance(cache, bool):
            cache_path = Path(cache)
            force_cache = True
        else:
            dirname = pathencode(repr(self.map_source))
            try:
                cache_path = make_cache_directory_path(
                    'map_downloader',
                    dirname
                )
            except FileNotFoundError:  # 存在可能性文件名过长？
                cache_path = make_cache_directory_path(
                    'map_downloader',
                    dhash(dirname).hexdigest()
                )
            force_cache = cache

        return force_cache, cache_path

    def download_tiles(self, tiles, level, cache_path):
        futures = []
        tile_source_index = 0
        progress = tqdm.tqdm(total=len(tiles))

        for x, y in tiles:
            futures.append(self.pool.apply_async(self.download_tile, (
                tile_source_index, x, y, level, cache_path
            )))
            tile_source_index = self.next_tile_source_index(tile_source_index)

        for (col, row), future in zip(tiles, futures):
            tile = future.get()
            yield col, row, tile

        progress.close()

    @staticmethod
    def encode_filename_with_level(path, level, levels, source):
        raw_path = os.fspath(path)
        if '{level}' in raw_path:
            path = Path(raw_path.format(level=level))
            if path.is_dir():
                path = path / f'{source}.png'
        else:
            path = Path(path)

            if len(levels) == 1:
                if path.is_dir():
                    path = path / f'{source}.png'
            else:
                if path.is_dir():
                    path = path / f'{source}.{level}.png'
                else:
                    example = path.with_suffix(f'.{{level}}{path.suffix}')
                    path = path.with_suffix(f'.{level}{path.suffix}')
                    warnings.warn(f'Multiple levels share same file name, renamed to `{path}`.'
                                  f' Try using a placeholder in path like `{example}` or path to a directory.')

        return path

    def next_tile_source_index(self, i):
        return (i + 1) % len(self.tile_locators)

    def download_tile(self, priority_source_index: int, col, row, level,
                      cache_path: Path | None = None):
        tile = None
        cache_file = None
        if cache_path is not None:
            cache_file = cache_path / f'{level}/{col}/{row}.png'

            if cache_file.exists() and os.path.getsize(cache_file) > 400:
                tile = Image.open(cache_file)
            else:
                ensure_filepath(cache_file)

        if tile is None:
            error_count = 0
            map_data = None
            while error_count < self.max_retries:
                priority_source_index = self.next_tile_source_index(priority_source_index)
                tile_source = self.tile_locators[priority_source_index]

                url = tile_source.query(col, row, level)

                try:
                    resp = self.session.get(url)
                    if resp.status_code == 200:
                        map_data = resp.content
                        break
                except requests.RequestException:
                    continue

                error_count += 1

            if map_data is None:
                raise RuntimeError(f'Failed to download map tile (Max retries {self.max_retries} exceeded).')

            tile = Image.open(BytesIO(map_data))

        if cache_path is not None:
            tile.save(cache_file)

        return tile

    @contextlib.contextmanager
    def activate_pool(self):
        self.running = True
        self.tile_locators = [t for t in self.map_source]
        self.session = requests.Session()

        try:
            yield
        finally:
            pass

        self.running = False
        self.tile_locators = None
        self.session = None
        self.progress = None
