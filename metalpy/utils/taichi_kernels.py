# 注意，不能使用from __future__ import annotations
# 会干扰taichi的类型系统，将ti.template等标注识别为字符串形式
# TODO: 关注taichi是否在该方面有相关更新

import taichi as ti

from .taichi import ti_func, ti_pyfunc, ti_kernel


@ti_pyfunc
def ti_use(_):
    pass


@ti_func
def ti_index(i, j, stype: ti.template(), struct_size, array_size):
    """对struct array的索引，返回对应的1d index

    Parameters
    ----------
    i
        第i个struct
    j
        struct的第j个元素
    stype
        struct的内存布局类型
    struct_size
        struct的大小
    array_size
        array的大小

    Returns
    -------
    ret
        1d索引
    """

    if ti.static(stype == ti.Layout.AOS):
        return i * struct_size + j  # [x, y, z, x, y, z, ..., x, y, z]
    else:
        return j * array_size + i  # [x, x, ..., x, y, y, ..., y, z, z, ..., z]


@ti_kernel
def from_sparse_coo(
        mat: ti.types.sparse_matrix_builder(),
        row: ti.types.ndarray(),
        col: ti.types.ndarray(),
        val: ti.types.ndarray(),
):
    for i in range(val.shape[0]):
        mat[row[i], col[i]] += val[i]


@ti_kernel
def from_sparse_coo(
        mat: ti.types.sparse_matrix_builder(),
        row: ti.types.ndarray(),
        col: ti.types.ndarray(),
        data: ti.types.ndarray(),
):
    for i in range(data.shape[0]):
        mat[row[i], col[i]] += data[i]


@ti_kernel
def from_sparse_csr(
        mat: ti.types.sparse_matrix_builder(),
        indptr: ti.types.ndarray(),
        indices: ti.types.ndarray(),
        data: ti.types.ndarray(),
):
    for i in range(indptr.shape[0] - 1):
        for j in range(indptr[i], indptr[i + 1]):
            mat[i, indices[j]] += data[j]


@ti_kernel
def from_sparse_csc(
        mat: ti.types.sparse_matrix_builder(),
        indptr: ti.types.ndarray(),
        indices: ti.types.ndarray(),
        data: ti.types.ndarray(),
):
    for i in range(indptr.shape[0] - 1):
        for j in range(indptr[i], indptr[i + 1]):
            mat[indices[j], i] += data[j]
