from typing import Sequence

from .array_ops import OperatorDispatcher, ArrayType

Concat = OperatorDispatcher('Concat').aliases({
    ArrayType.numpy: 'concatenate',
    ArrayType.pandas: 'concat',
})


def concat(arrays: Sequence, axis=0):
    """拼接数组

    Parameters
    ----------
    arrays
        数组列表
    axis
        拼接轴向

    Returns
    -------
    ret
        返回数组列表以对应方式拼接的结果
    """
    return Concat.dispatch(type(arrays[0]))(arrays, axis=axis)
