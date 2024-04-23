# 注意，不能使用from __future__ import annotations
# 会干扰taichi的类型系统，将ti.template等标注识别为字符串形式
# TODO: 关注taichi是否在该方面有相关更新

import taichi as ti

from .taichi import ti_func, ti_pyfunc, ti_kernel


@ti_pyfunc
def ti_use(
        _0,
        _1=0, _2=0, _3=0, _4=0, _5=0, _6=0, _7=0,
        _8=0, _9=0, _a=0, _b=0, _c=0, _d=0, _e=0,
        _f=0, _g=0, _h=0, _i=0, _j=0, _k=0, _l=0,
        _m=0, _n=0, _o=0, _p=0, _q=0, _r=0, _s=0,
        _t=0, _u=0, _v=0, _w=0, _x=0, _y=0, _z=0
):
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
