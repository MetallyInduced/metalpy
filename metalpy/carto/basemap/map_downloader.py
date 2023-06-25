from __future__ import annotations

import contextlib
import os
import warnings
from io import BytesIO
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Lock
from typing import Iterable, Generator

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
                 bounds=None,
                 mercator_bounds=None,
                 levels=(18,),
                 combine: bool | PathLike | None = None,
                 crop: bool | PathLike | None = True,
                 geotiff: PathLike | None = None,
                 cache: bool | PathLike | None = None) -> GeoImage:
        """下载区域内地图瓦片，并进行组合、裁剪得到参数指定区域的地图数据

        Parameters
        ----------
        west_lon, east_lon, south_lat, north_lat
            下载地图的西、东、南、北边界WGS 84坐标
        bounds
            下载地图的边界WGS 84坐标，以[lon_min, lon_max, lat_min, lat_max]顺序提供
        mercator_bounds
            下载地图的边界Pseudo-Mercator坐标，以[xmin, xmax, ymin, ymax]顺序提供
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
        cache缓存路径具体为'{cache_path}/{map_source_name}'
        """
        ret = None

        if not combine and not crop:
            assert geotiff, 'At least one of `combine`, `crop` or `geotiff` must be specified.'
            combine = True

        if bounds is not None:
            bounds = Bounds(bounds)
        elif mercator_bounds is not None:
            # 因为地图源的坐标转换以经纬度为单位，因此仍然需要一份WGS 84边界坐标
            mercator_bounds = Bounds(mercator_bounds)
            bounds = WebMercator.pseudo_mercator_to_wgs84(mercator_bounds[0:2], mercator_bounds[2:4])
            bounds = np.concatenate(bounds).view(Bounds)
        else:
            bounds = Bounds([west_lon, east_lon, south_lat, north_lat])

        if mercator_bounds is None:
            tmb_xy = WebMercator.wgs84_to_pseudo_mercator(bounds[0:2], bounds[2:4])
            true_mercator_bounds = np.concatenate(tmb_xy)
        else:
            true_mercator_bounds = np.asarray(mercator_bounds)
        true_mercator_bounds = Bounds(true_mercator_bounds)

        bounds.origin = self.map_source.warp_coordinates(bounds.origin)
        bounds.end = self.map_source.warp_coordinates(bounds.end)

        mb_xy = WebMercator.wgs84_to_pseudo_mercator(bounds[0:2], bounds[2:4])
        mercator_bounds = np.concatenate(mb_xy).view(Bounds)

        warped = not np.allclose(mercator_bounds, true_mercator_bounds)

        if not isinstance(levels, Iterable):
            levels = (levels,)

        with self.activate_pool():
            for level in levels:
                tiles_mercator_bounds = WebMercator.to_bounding_tiles_mercator_bounds(mercator_bounds, level)
                ref_system: GeoImageRefSystem | None = None
                tiles = self.download_geo_tiles(
                    mercator_bounds, level,
                    cache=cache
                )
                for tile in tiles:
                    if ref_system is None:
                        ref_system = tile.ref_system
                        if combine:
                            canvas = ref_system.create_image(
                                geo_bounds=tiles_mercator_bounds
                            )
                        else:  # 上文保证crop和combine至少有一个为True，因此此处crop=True
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

                # combine和crop输出png图像，与地理坐标系无关，因此坐标纠正可以在之后进行
                if warped:
                    # true_mercator_bounds为标准WGS 84坐标系下的地图边界
                    # mercator_bounds为地图源转换后坐标系下的地图边界
                    # 此处假设坐标转换前后区域尺度整体为线性变换，因此直接平移缩放转换到WGS 84下的地图边界
                    checks = []

                    # 尝试转换原点
                    ref_system.origin -= mercator_bounds.origin - true_mercator_bounds.origin

                    # 检查边界对齐情况
                    if crop:
                        edge_true_mercator_bounds = ref_system.to_edge_geo_bounds(true_mercator_bounds)
                        delta_bounds = np.abs(canvas.edge_geo_bounds - edge_true_mercator_bounds)
                        pixel_biases = delta_bounds.as_corners() / canvas.unit_size
                        if np.any(pixel_biases > 1):
                            checks.append(f'misaligned boundary ({pixel_biases.max():.2}px)')

                    # 计算源和目标区域缩放因子
                    scale_factor = true_mercator_bounds.extent / mercator_bounds.extent

                    # 检查区域尺度差异
                    relative_extent_diff = np.abs(scale_factor - 1)
                    if np.any(relative_extent_diff > 0.01):
                        checks.append(f'large scale difference ({relative_extent_diff.max():.2%})')

                    # 尝试转换坐标轴尺度
                    ref_system.unit_size *= scale_factor

                    if len(checks) > 0:
                        warnings.warn(f'{" and ".join(checks).capitalize()} detected.'
                                      f' due to coordinates transformation by `{self.map_source}`.'
                                      f' Consider using other map sources instead.')

                if geotiff:
                    geotiff_save_path = MapDownloader.encode_filename_with_level(
                        geotiff, level, levels, 'geotiff'
                    )

                    ret.save_geo_tiff(geotiff_save_path)

        return ret

    def download_geo_tiles(self,
                           mercator_bounds,
                           level,
                           cache: bool | PathLike | None = None,
                           true_tiles_mercator_bounds=None,
                           ref_system=None) -> Generator[GeoImage]:
        """下载Tile地图并转换为GeoTIFF形式

        Parameters
        ----------
        mercator_bounds
            Pseudo-Mercator边界
        level
            缩放等级
        cache
            缓存/缓存路径
        true_tiles_mercator_bounds
            边界坐标变换前的Pseudo-Mercator边界
        ref_system
            参考系（原点需为裁剪前瓦片地图区域左下角）

        Returns
        -------
        iterable[geo_tile]
            带坐标位置信息的Tile序列

        See Also
        --------
        MapDownloader.download : 地图下载主函数

        Notes
        -----
        某些地图源由于坐标系不一致，包含边界坐标变换步骤，需要调用者手动执行（参考MapDownloader.download方法）
        """
        tiles = list(WebMercator.iter_tiles(mercator_bounds, level))
        force_cache, cache_path = self.get_cache_path(cache)

        _cache_path = None
        if force_cache or len(tiles) > 100:
            _cache_path = cache_path

        tile_bounds = WebMercator.pseudo_mercator_bounds_to_tile(mercator_bounds, level)
        tiles_mercator_bounds = WebMercator.tile_bounds_to_pseudo_mercator(tile_bounds, level)

        if true_tiles_mercator_bounds is None:
            true_tiles_mercator_bounds = tiles_mercator_bounds
        else:
            true_tiles_mercator_bounds = Bounds(true_tiles_mercator_bounds)

        tile_extent = true_tiles_mercator_bounds.extent
        tile_count = tile_bounds.extent + 1
        col0, row1 = tile_bounds.xmin, tile_bounds.ymax

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
                    true_tiles_mercator_bounds.origin,
                    tile_extent / combined_image_size,
                    crs=WebMercator.PseudoMercator,
                    flip_y=True
                )

            geotile = ref_system.map_image(tile, offset=(
                (col - col0) * tile_size[0],
                (row1 - row) * tile_size[1]
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
            progress.update(1)
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
