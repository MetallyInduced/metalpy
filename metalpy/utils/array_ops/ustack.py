from typing import Literal

from . import is_array_like
from .array_ops import OperatorDispatcher, ArrayType

UStack = OperatorDispatcher('UStack')


def rstack(*arrays, ignore_index=True):
    """同 `np.r_` ，按行增长方向拼接
    """
    return ustack(*arrays, axis=0, ignore_index=ignore_index)


def cstack(*arrays, ignore_index=False):
    """同 `np.c_` ，按列增长方向拼接
    """
    return ustack(*arrays, axis=1, ignore_index=ignore_index)


def ustack(*arrays, axis: Literal[0, 1] = 0, ignore_index=False):
    """拼接向量或矩阵

    Parameters
    ----------
    arrays
        数组列表
    axis
        拼接轴向
    ignore_index
        忽略索引，主要用于pandas

    Returns
    -------
    ret
        返回获取的列，如果输入为切片范围或索引列表，则返回对应矩阵
    """
    assert axis in (0, 1), '`ustack` works only for axis 0 and 1 (as equivalence to np.r_ and np.c_).'
    if len(arrays) == 1:
        if not is_array_like(arrays[0]):
            arrays = tuple(arrays[0])

    return UStack.dispatch(type(arrays[0]))(tuple(arrays), axis=axis, ignore_index=ignore_index)


@UStack.register(ArrayType.numpy)
def _ustack_np(arrays, axis=0, ignore_index=False):
    import numpy as np
    if axis == 0:
        return np.r_.__getitem__(arrays)
    elif axis == 1:
        return np.c_.__getitem__(arrays)


@UStack.register(ArrayType.pandas)
def _ustack_pd(arrays, axis=0, ignore_index=False):
    import pandas as pd
    return pd.concat(arrays, axis=axis, ignore_index=ignore_index)
