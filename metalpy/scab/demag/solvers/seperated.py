import numpy as np
import psutil
import taichi as ti

from metalpy.scab.utils.misc import Field
from metalpy.utils.taichi import ti_kernel, ti_field, ti_FieldsBuilder, ti_func, copy_from
from .integrated import solve_Ax_b
from .kernel import kernel_matrix_forward
from .matrix_free import conjugate_gradient
from .solver import DemagnetizationSolver


class SeperatedSolver(DemagnetizationSolver):
    def __init__(
            self,
            receiver_locations: np.ndarray,
            xn: np.ndarray,
            yn: np.ndarray,
            zn: np.ndarray,
            base_cell_sizes: np.ndarray,
            source_field: Field,
            direct_to_host: bool,
            is_cpu: bool,
            progress: bool
    ):
        """该函数将矩阵分为9个部分，从而实现一些优化

        Parameters
        ----------
        receiver_locations
            观测点
        xn, yn, zn
            网格边界
        base_cell_sizes
            网格最小单元大小
        direct_to_host
            是否通过kernel将结果直接zero-copy地复制到待返回的numpy数组
        is_cpu
            是否是CPU架构，如果是，才会在内存可用时合并核矩阵然后求解
        progress
            是否输出求解进度条，默认为False不输出

        Notes
        -----
        优势：

        - 可以一定程度上绕过完整核矩阵尺寸的int32限制 (允许的网格规模为 `Integrated` 方法的3倍)
        - 可以使用AoS提高内存近邻性

        缺陷：

        - 网格规模仍然受到taichi的int32索引限制
        - 相比于 `Integrated` 方法，内存需求并不会降低，并且如果矩阵规模较小，计算效率可能会低于 `Integrated` 方法
        """
        super().__init__(receiver_locations, xn, yn, zn, base_cell_sizes, source_field)
        self.direct_to_host = direct_to_host
        self.is_cpu = is_cpu
        self.progress = progress

        self.builder = builder = ti_FieldsBuilder()

        nC = xn.shape[0]
        nObs = receiver_locations.shape[0]

        # [[Txx, Txy, Txz], [Tyx, Tyy, Tyz], [Tzx, Tzy, Tzz]]
        self.Tmat33 = [
            [ti_field(self.kernel_type) for _ in range(3)]
            for _ in range(3)
        ]
        self.Tmat9 = [t for ts in self.Tmat33 for t in ts]

        # 保证 Tx[xyz]，Ty[xyz]，Tz[xyz] 分别在空间上连续，提高空间近邻性
        for m in self.Tmat33:
            builder.dense(ti.ij, (nObs, nC)).place(*m)

        builder.finalize()

    def build_kernel(self, model):
        kernel_matrix_forward(
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.base_cell_sizes, model,
            *self.Tmat9, np.empty(0),
            write_to_mat=False, compressed=False
        )

    def solve(self, magnetization):
        if self.is_cpu and psutil.virtual_memory().percent < 45:
            # TODO: 进一步考虑模型大小来选择求解方案，模型较大时也应该使用 `solve_Tx_b`
            return solve_Ax_b(merge_Tmat_as_A(self.Tmat33, direct_to_host=self.direct_to_host), magnetization)
        else:
            return solve_Tx_b(self.Tmat33, magnetization, progress=self.progress)


def solve_Tx_b(Tmat33, m, progress: bool = False):
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

        conjugate_gradient(ti.linalg.LinearOperator(linear), b, x, progress=progress)

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
