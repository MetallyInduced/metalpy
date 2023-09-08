import pyvista as pv

from metalpy.carto.coords import Coordinates
from metalpy.utils.model import pv_ufunc_apply, DataSetLike


def warp_dataset(dataset: DataSetLike, src_crs, *, crs=None, query=None, inplace=False, return_crs=False):
    """转换数据集的坐标系

    Parameters
    ----------
    dataset
        待转换坐标系
    src_crs
        源坐标系
    crs
        目标坐标系
    query
        按指定模式搜索坐标系
    inplace
        指定转换是否就地操作
    return_crs
        指示时候返回采用的新坐标系，常用与配合`query`，获取查询到的坐标系信息

    Returns
    -------
    result
        转换坐标系后的数据集，如果指定了`return_crs=True`，则额外返回转换后的坐标系信息

    See Also
    --------
    :py:class:`metalpy.carto.coords.Coordinates` : 用于表示与转换坐标
    """
    dest_crs = [None]

    def wrapper(d):
        if dest_crs[0] is None:
            _, dest_crs[0] = _warp_dataset(d, src_crs, crs=crs, query=query, return_crs=True)
        else:
            _ = _warp_dataset(d, src_crs, crs=dest_crs[0])

    warped = pv_ufunc_apply(
        dataset, wrapper,
        inplace=inplace
    )

    if return_crs:
        return warped, dest_crs[0]
    else:
        return warped


def _warp_dataset(dataset: pv.DataSet, src_crs, *, crs=None, query=None, return_crs=False):
    geo_points: Coordinates = dataset.points.view(Coordinates).with_crs(src_crs)

    dest_crs = None
    if crs is not None or query is not None:
        geo_points.warp(crs=crs, query=query, inplace=True)
        try:
            dataset.points = geo_points
        except AttributeError:
            # 例如RectilinearGrid这类参数化对象无法直接设置模型点位置
            # 转换为UnstructuredGrid再进行转换
            dataset = dataset.cast_to_unstructured_grid()
            dataset.points = geo_points

        if return_crs:
            dest_crs = geo_points.crs

    if return_crs:
        return dataset, dest_crs
    else:
        return dataset
