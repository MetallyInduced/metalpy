import numpy as np
from PIL import Image
from pyproj import CRS

from metalpy.utils.bounds import Bounds


class GeoImageRefSystem:
    def __init__(self, origin, unit_size, crs=None, flip_y=None):
        """用于表示地理图像的各个像素中心点和地理坐标的映射关系

        Parameters
        ----------
        origin
            地理图像参考系的原点，以像素边缘为边界
        unit_size
            地理图像参考系每个单元（像素）对应的地理距离
        crs
            地理参考坐标系
        flip_y
            指示地理坐标系与图像坐标系的y轴方向是否相反

        Notes
        -----
        地理坐标系保证与单位网格坐标系方向相同，图像坐标系只在涉及图像处理时才会引入，此时才会考虑`flip_y`等问题
        """
        unit_size = np.array(unit_size)
        origin = np.array(origin)

        assert unit_size[0] > 0, 'Flipping of X axis is not allowed.'
        if flip_y is None:
            flip_y = unit_size[1] < 0

        self.origin = origin
        self.unit_size = np.abs(unit_size)
        self.flip_y = flip_y

        if crs is not None:
            if isinstance(crs, str):
                crs = CRS.from_string(crs)
            elif isinstance(crs, int):
                crs = CRS.from_epsg(crs)
            self.crs = crs
        else:
            self.crs = CRS.from_string('WGS 84')

    def create_image(self, image_size=None, offset=(0, 0), bounds=None, geo_bounds=None):
        from metalpy.carto.basemap.geo_image import GeoImage

        if bounds is not None:
            bounds = Bounds(bounds)
            offset = bounds.origin
            image_size = bounds.extent
        elif geo_bounds is not None:
            bounds = self.to_unit_bounds(geo_bounds)
            offset = bounds.origin
            image_size = bounds.extent
        else:
            image_size = np.asarray(image_size)
            offset = np.asarray(offset)

        nx, ny = image_size.astype(np.intp)
        offset = offset.astype(np.intp)

        return GeoImage(Image.new('RGBA', (nx, ny)), offset, self)

    def map_image(self, image: Image.Image, offset=(0, 0), geo_offset=None):
        from metalpy.carto.basemap.geo_image import GeoImage

        if geo_offset is not None:
            offset = self.to_unit_bounds(np.c_[geo_offset, (0, 0)].ravel())

        return GeoImage(image, offset, self)

    def to_unit_bounds(self, bounds):
        corners = (Bounds(bounds).as_corners() - self.origin) / self.unit_size
        corners.origin = np.floor(corners.origin)
        corners.end = np.ceil(corners.end)
        return corners.as_bounds()

    def to_geo_bounds(self, bounds):
        corners = Bounds(bounds).as_corners() * self.unit_size + self.origin
        return corners.as_bounds()

    def to_edge_geo_bounds(self, geo_bounds):
        return self.to_geo_bounds(self.to_unit_bounds(geo_bounds))

    @staticmethod
    def of_unit_edge_bounds(edge_origin, unit_size, crs=None, flip_y=None) -> 'GeoImageRefSystem':
        """
        Notes
        -----
                                 ↓ end[right top]
            ┌────┬────┬─────┬────┐ ←
            │    │    │ ... │    │
            ├────┼────┼─────┼────┤
            │    │    │ ... │    │
            ├────┼────┼─────┼────┤
            │    │    │ ... │    │
          → └────┴────┴─────┴────┘
            ↑ origin[left bottom]
        """
        return GeoImageRefSystem(edge_origin, unit_size, crs=crs, flip_y=flip_y)

    @staticmethod
    def of_unit_center_bounds(center_origin, unit_size, crs=None, flip_y=None) -> 'GeoImageRefSystem':
        """
        Notes
        -----
                              ↓ end[right top]
            ┌────┬────┬─────┬────┐
            │    │    │ ... │    │ ←
            ├────┼────┼─────┼────┤
            │    │    │ ... │    │
            ├────┼────┼─────┼────┤
          → │    │    │ ... │    │
            └────┴────┴─────┴────┘
              ↑ origin[left bottom]
        """
        return GeoImageRefSystem(
            center_origin - np.asarray(unit_size) / 2,
            unit_size,
            crs=crs,
            flip_y=flip_y
        )
