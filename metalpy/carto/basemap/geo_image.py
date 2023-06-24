from __future__ import annotations

import numpy as np
from PIL.Image import Image
import PIL.Image

from metalpy.utils.bounds import Bounds
from metalpy.carto.coords import Coordinates
from metalpy.carto.utils.crs import CRSLike
from metalpy.carto.coords.coordinates import CRSQuery

from .geo_image_ref_system import GeoImageRefSystem


class GeoImage:
    def __init__(self,
                 image: Image,
                 offset,
                 ref_system: GeoImageRefSystem):
        """包含地理坐标信息的图像

        Parameters
        ----------
        image
            原始图像
        offset
            图像原点到坐标系原点的单元距离，图像原点定义在上或在下与`ref_system.flip_y`有关
        ref_system
            参考坐标系

        Notes
        -----
        当`flip_y=True`，地理坐标系和图像坐标系的y轴方向相反，如图所示
        此时offset为从坐标系原点到图像左下角的距离
        ↑ unit_y (m)
        │  ┌────────────────────────→ img_x (px)
        │  │┌────────────────────┐
        │  ││                    │
        │  ││          ┌─────┐   │
        │  ││          │     │   │
        │  ││          └─────┘   │
        │  │└────────────────────┘
        │  ↓ img_y (px)
        └───────────────────────────────────→ unit_x (m)

        当`flip_y=False`，地理坐标系和图像坐标系的y轴方向相同，如图所示
        此时offset为从坐标系原点到图像左上角的距离
        ↑ unit_y (m)
        │  ↑ img_y (px)
        │  │┌────────────────────┐
        │  ││                    │
        │  ││          ┌─────┐   │
        │  ││          │     │   │
        │  ││          └─────┘   │
        │  │└────────────────────┘
        │  └────────────────────────→ img_x (px)
        └───────────────────────────────────→ unit_x (m)
        """
        self.image = image
        self.offset = np.array(offset)
        self.ref_system = ref_system

    @property
    def unit_size(self):
        return self.ref_system.unit_size

    @property
    def size(self):
        return self.image.size

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def bounds(self):
        return np.c_[
            self.offset,
            self.offset + self.size
        ].ravel().view(Bounds)

    @property
    def geo_bounds(self):
        """边界像素的中心点所形成的边界框
        """
        return self.ref_system.to_geo_bounds(
            self.bounds
            + (0.5, -0.5, 0.5, -0.5)
        )

    @property
    def edge_geo_bounds(self):
        """边界像素的外边界框
        """
        return self.ref_system.to_geo_bounds(
            self.bounds
        )

    @property
    def flip_y(self):
        return self.ref_system.flip_y

    @property
    def crs(self):
        return self.ref_system.crs

    @property
    def mode(self):
        return self.image.mode

    def crop(self, bounds=None, geo_bounds=None):
        if bounds is None:
            assert geo_bounds is not None, 'Either one of `bounds` or `geo_bounds` is required.'
            bounds = self.ref_system.to_unit_bounds(geo_bounds)

        corners = bounds.as_corners()
        offset = corners.origin
        rel_corners = corners - self.bounds.as_corners().origin
        if self.flip_y:
            rel_corners.origin[1], rel_corners.end[1] = (
                self.height - rel_corners.end[1],
                self.height - rel_corners.origin[1]
            )

        return GeoImage(
            image=self.image.crop(rel_corners.ravel()),
            offset=offset,
            ref_system=self.ref_system
        )

    def paste(self, img: 'GeoImage'):
        """将新地理图像`img`粘贴到当前图像（会修改当前图像）

        Parameters
        ----------
        img
            待粘贴的地理图像
        """
        if self.ref_system.flip_y:
            offset = (
                img.bounds[0] - self.bounds[0],
                self.bounds[3] - img.bounds[3]
            )
        else:
            offset = self.bounds[[0, 2]] - img.bounds[[0, 2]]

        self.image.paste(img.image, offset)

    def to_polydata(self, dest_crs: CRSLike | None = None, query_dest_crs: CRSQuery | None = None):
        import pyvista as pv

        bounds = self.edge_geo_bounds
        xs = np.linspace(bounds.xmin, bounds.xmax, self.width + 1)
        ys = np.linspace(bounds.ymin, bounds.ymax, self.height + 1)

        if self.flip_y:
            ys = ys[::-1]

        grid = pv.RectilinearGrid(xs, ys, 0)

        img = np.asarray(self.image)
        n_channels = img.shape[2] if img.ndim > 2 else 1
        grid['Image'] = img.reshape(-1, n_channels)

        grid = grid.cast_to_unstructured_grid()
        geo_points: Coordinates = grid.points.view(Coordinates).with_crs(self.crs)

        if dest_crs is not None or query_dest_crs is not None:
            geo_points.warp(crs=dest_crs, query=query_dest_crs, inplace=True)
            grid.points = geo_points

        return grid

    def save(self, fp, format=None, **params):
        self.image.save(fp, format=format, **params)

    def save_geo_tiff(self, path):
        import rasterio
        from rasterio.transform import Affine

        # 需要注意图像矩阵存储向下为正，而数据向上（北）为正。
        # offset为左上角像素中心点位置，左上角实际为(xmin, ymax)，
        # 且y_scale < 0，即图像行索引增加方向为坐标值减少方向（南方向）
        bounds = self.geo_bounds
        transform = Affine.translation(bounds.xmin, bounds.ymax) * Affine.scale(self.unit_size[0], -self.unit_size[1])
        image_arr = np.asarray(self.image)

        if image_arr.ndim < 3:
            channels = 1
        else:
            channels = image_arr.shape[2]

        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            width=self.width,
            height=self.height,
            count=channels,
            dtype=image_arr.dtype,
            crs=self.crs,
            transform=transform,
        ) as dataset:
            if image_arr.ndim < 3:
                dataset.write(image_arr, 1)
            else:
                for dim in range(channels):
                    dataset.write(image_arr[:, :, dim], dim + 1)

    @staticmethod
    def read(path):
        import rasterio
        from rasterio.plot import reshape_as_image
        from affine import Affine

        with rasterio.open(path) as dataset:
            transform = dataset.transform
            assert transform.identity() == Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), \
                'GeoImage supports only translation and scaling as transform.'

            origin = (transform.xoff, transform.yoff)
            unit_size = (transform[0], transform[4])
            ref = GeoImageRefSystem.of_unit_center_bounds(
                center_origin=origin,
                unit_size=unit_size,
                crs=dataset.crs
            )
            raster = reshape_as_image(dataset.read())
            image = PIL.Image.fromarray(raster)

            if not ref.flip_y:
                offset = (0, 0)
            else:
                offset = (0, -image.height)

            return ref.map_image(image, offset=offset)
