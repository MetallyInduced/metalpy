import pyvista as pv

from metalpy.carto.basemap import MapDownloader
from metalpy.carto.basemap.sources import AMapSource
from metalpy.carto.coords import Coordinates
from metalpy.scab.modelling import Scene
from metalpy.utils.bounds import Bounds


def main():
    dest_crs = Coordinates.SearchUTM
    bounds = Bounds([121.49238, 121.49824, 31.24027, 31.24345])  # 东方明珠

    scene = Scene.from_osm(bounds, dest_crs=dest_crs, default_height=10)  # OSM中未标注高度的建筑会指定高度为default_height
    basemap = MapDownloader(AMapSource(satellite=True)).download(bounds=bounds.expand(proportion=0.1), cache=True)

    del scene.objects_layer.objects[0]  # 移除东方明珠塔的建筑外表面
    for i, s in enumerate(scene):
        s.model = i

    p = pv.Plotter()
    p.add_mesh(scene.to_multiblock(), opacity=0.5)
    p.add_mesh(basemap.to_polydata(query_dest_crs=dest_crs))
    p.show()


if __name__ == '__main__':
    main()
