Metalpy Carto
===========================

**Metalpy Carto** provides functionalities related to cartography, which only supports downloading
basemap for now.


Installation
------------
**Metalpy Carto** is now a submodule in metalpy, which can be installed using pip:

```console
pip install metalpy[carto]
```

Features
--------
1. Download basemap from public tile map sources including Google Map, Wayback and others.
and convert to **geotiff** for further use like **GeoServer**. 
```python
from metalpy.carto.basemap import MapDownloader
from metalpy.carto.basemap.sources import GoogleMapSource

if __name__ == '__main__':
    source = GoogleMapSource(satellite=True, labels=True)
    map_downloader = MapDownloader(source)
    map_downloader.download(
        bounds=[139.278433, 140.506452, 35.968355, 35.427143],
        levels=(11, 11),
        crop='./basemap.png',  # crop the image to match the bounds
        geotiff='./geotiff.tif'  # convert to geotiff format
    )
```

2. Download maps from custom map sources (e.g. ArcGIS Online). Supported placeholders includes:

|     Type      |  Placeholder  |
|:-------------:|:-------------:|
|     Scale     | {level} / {z} |
|   Tile Col    |  {col} / {x}  |
|   Tile Row    |  {row} / {y}  |

Take [`World Imagery (Wayback 2023-05-03)`](https://esri.maps.arcgis.com/home/item.html?id=f95ee415e16145e4b70bf10e7a4dd6f5)
for example, url template can be found at the right bottom of the overview page (as of 2023/6/11).
```python
from metalpy.carto.basemap import MapDownloader
from metalpy.carto.basemap.sources import SimpleTileMapSource

if __name__ == '__main__':
    # '{' and '}' may be encoded as '%7B' and '%7D', convert it back.
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
- [ ] Support standard WMTS source
