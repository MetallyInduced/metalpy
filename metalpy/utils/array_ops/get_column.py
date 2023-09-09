from .array_ops import OperatorDispatcher, ArrayType

GetColumn = OperatorDispatcher('GetColumn')


def get_column(arr, ind):
    """从数组获取列

    Parameters
    ----------
    arr
        数组实例
    ind
        列索引或列索引数组 / 切片slice / 列名或列名数组（pandas）

    Returns
    -------
    ret
        返回获取的列，如果输入为切片范围或索引列表，则返回对应矩阵
    """
    return GetColumn.dispatch(type(arr))(arr, ind)


@GetColumn.register(ArrayType.numpy)
def _get_column_np(arr, ind):
    return arr[:, ind]


@GetColumn.register(ArrayType.pandas)
def _get_column_pd(arr, ind):
    if not isinstance(ind, slice):  # slice直接索引会默认索引行，需要排除
        try:
            return arr[ind]  # 判断是否为合法的列key
        except KeyError:
            pass

    return arr.iloc[:, ind]  # 否则认为是列索引
