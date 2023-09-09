import numpy as np

from .array_ops import OperatorDispatcher, ArrayType

GetColumnsExcept = OperatorDispatcher('GetColumnsExcept')


def get_columns_except(arr, ind):
    """从数组获取除去指定列外的部分

    Parameters
    ----------
    arr
        数组实例
    ind
        列索引或列索引数组 / 切片slice / 列名或列名数组（pandas）

    Returns
    -------
    ret
        返回不在ind中指定的列
    """
    return GetColumnsExcept.dispatch(type(arr))(arr, ind)


def _excluded_mask(length, excludes):
    take = np.ones(length, dtype=bool)
    take[excludes] = False

    return take


@GetColumnsExcept.register(ArrayType.numpy)
def _get_columns_except_np(arr, ind):
    return arr[:, _excluded_mask(arr.shape[1], ind)]


@GetColumnsExcept.register(ArrayType.pandas)
def _get_columns_except_pd(arr, ind):
    if not isinstance(ind, slice):  # slice直接索引会默认索引行，需要排除
        try:
            _ = arr[ind]  # 判断是否为合法的列key
            keys = set(arr.keys())
            keys.difference_update(ind)
            return arr[list(keys)]
        except KeyError:
            pass

    return arr.iloc[:, _excluded_mask(arr.shape[1], ind)]  # 否则认为是列索引
