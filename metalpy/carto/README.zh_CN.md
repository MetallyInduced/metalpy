Metalpy Carto
===========================

**Metalpy Carto** 提供制图相关功能，目前仅包含下载地图。


Installation
------------
**Metalpy Carto** 目前是metalpy的一个子模块，你可以使用pip安装它：

```console
pip install metalpy[carto]
```

Features
--------
1. 从公开瓦片地图数据源下载底图，支持包括谷歌底图、Wayback历史影像在内的多个数据源。
支持转换为**geotiff**格式以适用于**GeoServer**等应用。
```python
from metalpy.carto.basemap import MapDownloader
from metalpy.carto.basemap.sources import GoogleMapSource

if __name__ == '__main__':
    source = GoogleMapSource(satellite=True, labels=True)
    map_downloader = MapDownloader(source)
    map_downloader.download(
        bounds=[139.278433, 140.506452, 35.427143, 35.968355],
        levels=11,
        crop='./basemap.png',  # crop the image to match the bounds
        geotiff='./geotiff.tif'  # convert to geotiff format
    )
```

2. 从自定义数据源进行地图下载（如ArcGIS Online），支持的占位符包括：

|   类型   |      占位符      |
|:------:|:-------------:|
|  缩放级别  | {level} / {z} |
| 东西方向索引 |  {col} / {x}  |
| 南北方向索引 |  {row} / {y}  |

例如下载[`World Imagery (Wayback 2023-05-03)`](https://esri.maps.arcgis.com/home/item.html?id=f95ee415e16145e4b70bf10e7a4dd6f5)，
数据源链接可以在网页右下角找到（截至2023/6/11有效）。
```python
from metalpy.carto.basemap import MapDownloader
from metalpy.carto.basemap.sources import SimpleTileMapSource

if __name__ == '__main__':
    # 网页原始链接可能将'{'和'}'编码为了'%7B'和'%7D'，需要手动转换
    source = SimpleTileMapSource('https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/46399/{level}/{row}/{col}')
    map_downloader = MapDownloader(source)
    map_downloader.download(
        bounds=[139.278433, 140.506452, 35.427143, 35.968355],
        levels=11,
        crop='./basemap.png',  # crop the image to match the bounds
        geotiff='./geotiff.tif'  # convert to geotiff format
    )
```

TODOs
-----
- [ ] 支持标准WMTS地图源
