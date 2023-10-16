import numpy as np
import psutil
import taichi as ti

from metalpy.utils.taichi import ti_kernel, ti_field, ti_FieldsBuilder, ti_func, copy_from
from .integrated import solve_Ax_b
from .kernel import kernel_matrix_forward


def forward_seperated(
        magnetization: np.ndarray,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        base_cell_sizes: np.ndarray,
        susc_model: np.ndarray,
        direct_to_host: bool,
        is_cpu: bool,
        verbose: bool
):
    """该函数将矩阵分为9个部分，从而实现一些优化

    Parameters
    ----------
    magnetization
        磁化强度
    receiver_locations
        观测点
    xn, yn, zn
        网格边界
    base_cell_sizes
        网格最小单元大小
    susc_model
        磁化率模型
    direct_to_host
        是否通过kernel将结果直接zero-copy地复制到待返回的numpy数组
    is_cpu
        是否是CPU架构，如果是，才会在内存可用时合并核矩阵然后求解
    verbose
        是否输出额外的日志信息，例如taichi的线性求解器的进度信息，True则输出

    Notes
    -----
        优势：
        1. 可以一定程度上绕过完整核矩阵尺寸的int32限制 (允许的网格规模为 `Integrated` 方法的3倍)
        2. 可以使用AoS提高内存近邻性

        缺陷：
        1. 网格规模仍然受到taichi的int32索引限制
        2. 相比于 `Integrated` 方法，内存需求并不会降低，并且如果矩阵规模较小，计算效率可能会低于 `Integrated` 方法
    """
    with ti_FieldsBuilder() as builder:
        nC = xn.shape[0]
        nObs = receiver_locations.shape[0]

        # [[Txx, Txy, Txz], [Tyx, Tyy, Tyz], [Tzx, Tzy, Tzz]]
        Tmat33 = [
            [ti_field(ti.f64) for _ in range(3)]
            for _ in range(3)
        ]
        Tmat9 = [t for ts in Tmat33 for t in ts]

        # 保证 Tx[xyz]，Ty[xyz]，Tz[xyz] 分别在空间上连续，提高空间近邻性
        for m in Tmat33:
            builder.dense(ti.ij, (nObs, nC)).place(*m)

        builder.finalize()

        kernel_matrix_forward(receiver_locations, xn, yn, zn, base_cell_sizes, susc_model,
                              *Tmat9, np.empty(0), write_to_mat=False, compressed=False)

        if is_cpu and psutil.virtual_memory().percent < 45:
            # TODO: 进一步考虑模型大小来选择求解方案，模型较大时也应该使用 `solve_Tx_b`
            return solve_Ax_b(merge_Tmat_as_A(Tmat33, direct_to_host=direct_to_host), magnetization)
        else:
            return solve_Tx_b(Tmat33, magnetization, quiet=not verbose)


def solve_Tx_b(Tmat33, m, quiet: bool = True):
    with ti_FieldsBuilder() as builder:
        x = builder.place_dense_like(m[:, None])
        b = builder.place_dense_like(m[:, None])

        builder.finalize()

        copy_from(b, m[:, None])

        @ti_func
        def mul3(Ax, x, i: ti.template()):
            multiply_with_row_stride_3(
                Ax, *Tmat33[i], x, i, 3
            )

        @ti_kernel
        def linear(x: ti.template(), Ax: ti.template()):
            mul3(Ax, x, 0)
            mul3(Ax, x, 1)
            mul3(Ax, x, 2)

        # TODO: 添加求解进度条
        ti.linalg.taichi_cg_solver(ti.linalg.LinearOperator(linear), b, x, quiet=quiet)

        return x.to_numpy()


def merge_Tmat_as_A(Tmat33, direct_to_host=True):
    nObs, nC = Tmat33[0][0].shape
    Amat = np.empty((3 * nObs, 3 * nC), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if direct_to_host:
                tensor_to_ext_arr(Tmat33[i][j], Amat, i, 3, j, 3)
            else:
                Amat[i::3, j::3] = Tmat33[i][j].to_numpy()

    return Amat


@ti_kernel
def tensor_to_ext_arr(tensor: ti.template(), arr: ti.types.ndarray(),
                      x0: ti.template(), xstride: ti.template(),
                      y0: ti.template(), ystride: ti.template()):
    for I in ti.grouped(tensor):
        arr[I[0] * xstride + x0, I[1] * ystride + y0] = tensor[I]


@ti_func
def multiply_with_row_stride_3(
        out: ti.template(),
        A: ti.template(),
        B: ti.template(),
        C: ti.template(),
        v: ti.template(),
        row0: ti.template(), row_stride: ti.template()
):
    for r in range(A.shape[0]):
        row = r * row_stride + row0

        summed = (
                A[r, 0] * v[0, 0]
                + B[r, 0] * v[1, 0]
                + C[r, 0] * v[2, 0]
        )

        for c in range(1, A.shape[1]):
            col = c * 3
            summed += (
                    A[r, c] * v[col + 0, 0]
                    + B[r, c] * v[col + 1, 0]
                    + C[r, c] * v[col + 2, 0]
            )

        out[row, 0] = summed
