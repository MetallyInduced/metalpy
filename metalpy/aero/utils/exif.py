from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import tqdm
from PIL import Image, UnidentifiedImageError

from metalpy.utils.file import file_cached, PathLike, PathLikeType


@file_cached
def load_gps_info_from_exifs(paths: Sequence[PathLike] | PathLike):
    """从图像EXIF中读取位置数据信息

    Parameters
    ----------
    paths
        图像文件或所在文件夹路径，支持通过序列指定多个路径

    Returns
    -------
    exifs
        各个图像的EXIF数据
    """
    import piexif
    import pandas as pd

    if isinstance(paths, PathLikeType):
        paths = [paths]

    files_to_load = []
    for path in paths:
        if os.path.isfile(path):
            files_to_load.append(Path(path))
        else:
            for base, _, files in os.walk(path):
                base = Path(base)
                for file in files:
                    files_to_load.append(base / file)

    pts = []
    for p in tqdm.tqdm(files_to_load):
        try:
            img = Image.open(p)
        except UnidentifiedImageError:
            continue

        # TODO: other data? elevation?
        info = piexif.load(img.info['exif'])
        long = convert_rational_number(info['GPS'][piexif.GPSIFD.GPSLongitude], 60)
        long *= convert_gps_ref_code_to_sign(info['GPS'][piexif.GPSIFD.GPSLongitudeRef])
        lat = convert_rational_number(info['GPS'][piexif.GPSIFD.GPSLatitude], 60)
        lat *= convert_gps_ref_code_to_sign(info['GPS'][piexif.GPSIFD.GPSLatitudeRef])
        alt = convert_rational_number(info['GPS'][piexif.GPSIFD.GPSAltitude])
        alt *= convert_gps_ref_code_to_sign(info['GPS'][piexif.GPSIFD.GPSAltitudeRef])
        pts.append({
            'Longitude': long,
            'Latitude': lat,
            'Altitude': alt
        })

    return pd.DataFrame(pts)


def convert_rational_number(num_list, unit=10):
    if not isinstance(num_list, Sequence):
        return num_list
    if not isinstance(num_list[0], Sequence):
        num_list = (num_list,)

    p = 1
    ret = 0
    for a, b in num_list:
        ret += a / b / p
        p *= unit

    return ret


def convert_gps_ref_code_to_sign(code: str | int):
    if isinstance(code, str):
        code = code.upper()
    if code in {'E', 'N', b'E', b'N', 0}:
        return 1
    else:
        return -1
