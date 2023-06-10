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

Some additional functions, like creating **geotiff** image, require GDAL to work properly,
it can be installed with conda or by following instructions from
[GDAL](https://gdal.org/download.html).
```console
conda install libgdal
```

Features
--------
1. Download basemap from public tile map sources including Google Map, Wayback and others.
and convert to **geotiff** for further use like **GeoServer** (requires GDAL). 
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

TODOs
-----
- [ ] Implement coordinate transformation.
- [ ] Import basemap as 3D objects.
- [ ] Implement Wayback source.
