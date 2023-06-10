import os
import warnings
from pathlib import Path
from shutil import which


def check_gdal(cmd_name):
    if 'PROJ_LIB' not in os.environ:
        # 尝试修复找不到proj.db的问题
        # https://github.com/PDAL/PDAL/issues/2544

        # 如果是conda安装，有可能设置了GDAL_DATA但PROJ_LIB未设置
        if 'GDAL_DATA' in os.environ:
            # proj包和libgdal包同在`${env}/Library/share`目录下
            proj_lib_path = Path(os.environ['GDAL_DATA']) / '../proj'
            proj_lib_path = proj_lib_path.resolve()

            if proj_lib_path.exists():
                config_gdal(proj_lib=proj_lib_path)
            else:
                warnings.warn('`PROJ_LIB` not found in environment variables.'
                              ' Please check your `GDAL` installation.')
        else:
            warnings.warn('`GDAL_DATA` not found in environment variables.'
                          ' Please check your `GDAL` installation.')

    ret = which(cmd_name)
    if ret is None:
        raise RuntimeError('`GDAL` not found in PATH.'
                           ' `GDAL` can be installed with'
                           ' `conda install libgdal`'
                           ' or by following instructions from'
                           ' `https://gdal.org/download.html`.'
                           ' Or check to ensure `GDAL` binaries in PATH.')

    return ret


def config_gdal(proj_lib, gdal_data=None):
    os.environ['PROJ_LIB'] = os.fspath(proj_lib)
    if gdal_data is not None:
        os.environ['GDAL_DATA'] = os.fspath(gdal_data)
