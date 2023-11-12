from .array_ops import OperatorDispatcher, ArrayType

GetRow = OperatorDispatcher('GetRow')


def get_row(arr, ind):
    """从数组获取行

    Parameters
    ----------
    arr
        数组实例
    ind
        行索引或行索引数组 / 切片slice / 行索引或行索引数组（pandas）

    Returns
    -------
    ret
        返回获取的行，如果输入为切片范围或索引列表，则返回对应矩阵

    Notes
    -----
    对于Pandas，直接传入整数序列或切片时默认按行号索引，如果需要按索引取值，则需要通过 `pd.Index` 包装
    """
    return GetRow.dispatch(type(arr))(arr, ind)


@GetRow.register(ArrayType.numpy)
def _get_row_np(arr, ind):
    return arr[ind]


@GetRow.register(ArrayType.pandas)
def _get_row_pd(arr, ind):
    import pandas as pd
    if isinstance(ind, pd.Index) or getattr(ind, 'dtype', None) == bool:
        return arr.loc[ind]  # 行索引
    else:
        return arr.iloc[ind]  # 否则认为是行号
