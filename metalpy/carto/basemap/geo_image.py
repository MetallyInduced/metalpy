import os
import subprocess
from pathlib import Path

import numpy as np
from PIL.Image import Image

from metalpy.utils.bounds import Bounds
from metalpy.utils.file import make_cache_file_path
from .gdal_helper import check_gdal
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

        Notes
        -----

        Notes
        -----
        当`flip_y=True`，地理坐标系和图像坐标系的y轴方向相反，如图所示
        此时offset为从坐标系原点到图像左下角的距离
        ↑ unit_y
        │  ┌────────────────────────→ img_x
        │  │┌────────────────────┐
        │  ││                    │
        │  ││          ┌─────┐   │
        │  ││          │     │   │
        │  ││          └─────┘   │
        │  │└────────────────────┘
        │  ↓ img_y
        └───────────────────────────────────→ unit_x

        当`flip_y=False`，地理坐标系和图像坐标系的y轴方向相同，如图所示
        此时offset为从坐标系原点到图像左上角的距离
        ┌──────────────────────────────────→ unit_x
        │  ┌────────────────────────→ img_x
        │  │┌────────────────────┐
        │  ││                    │
        │  ││          ┌─────┐   │
        │  ││          │     │   │
        │  ││          └─────┘   │
        │  │└────────────────────┘
        │  ↓ img_y
        ↓ unit_y
        """
        self.image = image
        self.offset = np.array(offset)
        self.ref_system = ref_system

    @property
    def unit_size(self):
        return self.image.size

    @property
    def unit_width(self):
        return self.unit_size[0]

    @property
    def unit_height(self):
        return self.unit_size[1]

    @property
    def unit_bounds(self):
        return np.c_[
            self.offset,
            self.offset + self.image.size
        ].ravel().view(Bounds)

    @property
    def geo_bounds(self):
        return self.ref_system.to_geo_bounds(
            self.unit_bounds
            + (0.5, -0.5, 0.5, -0.5)
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
        rel_corners = corners - self.unit_bounds.as_corners().origin
        if self.flip_y:
            rel_corners.origin[1], rel_corners.end[1] = (
                self.unit_height - rel_corners.end[1],
                self.unit_height - rel_corners.origin[1]
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
                img.unit_bounds[0] - self.unit_bounds[0],
                self.unit_bounds[3] - img.unit_bounds[3]
            )
        else:
            offset = self.unit_bounds[[0, 2]] - img.unit_bounds[[0, 2]]

        self.image.paste(img.image, offset)

    def save(self, fp, format=None, **params):
        self.image.save(fp, format=format, **params)

    def apply_geo_info(self,
                       input_file,
                       output_file,
                       bounds_transform: callable = None):
        gdal_translate = check_gdal('gdal_translate')

        bounds = self.geo_bounds
        if bounds_transform is not None:
            bounds = bounds_transform(bounds)
        bounds = Bounds(bounds).as_corners().ravel().astype(str)

        subprocess.check_call([
            gdal_translate,
            f'-of', 'GTiff',
            f'-a_srs', self.crs.to_string(),
            f'-a_ullr', *bounds,
            os.fspath(input_file),
            os.fspath(output_file)
        ])

    def save_geo_tiff(self, path):
        path = Path(path)
        cache_img = make_cache_file_path('geoimage', path.with_suffix(f'.temp{path.suffix}'))
        self.save(cache_img)
        self.apply_geo_info(input_file=cache_img, output_file=path)
