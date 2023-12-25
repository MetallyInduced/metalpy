import numpy as np
import psutil
import taichi as ti
from taichi.lang.util import to_taichi_type

from metalpy.scab.utils.misc import Field
from metalpy.utils.taichi import ti_kernel, ti_field, ti_FieldsBuilder, ti_func
from metalpy.utils.ti_solvers import matrix_free
from .integrated import solve_Ax_b
from .kernel import kernel_matrix_forward
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
            kernel_dtype=None,
            progress: bool = False,
            direct_to_host: bool = False
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
        source_field
            定义默认外部场源，求解时若未指定场源，则使用该值
        kernel_dtype
            核矩阵数据类型，默认为None，自动从输入数据推断
        direct_to_host
            是否通过kernel将结果直接zero-copy地复制到待返回的numpy数组
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
        super().__init__(receiver_locations, xn, yn, zn, base_cell_sizes, source_field, kernel_dtype, progress)
        self.direct_to_host = direct_to_host

        self.builder = builder = ti_FieldsBuilder()

        self.Tmat321 = [
            [ti_field(self.kernel_dtype) for _ in range(3)],  # Txx, Txy, Txz
            [ti_field(self.kernel_dtype) for _ in range(2)],  # ---- Tyy, Tyz
            [ti_field(self.kernel_dtype) for _ in range(1)],  # --------- Tzz
        ]
        self.Tmat33 = [
            self.Tmat321[0],  # Txx, Txy, Txz
            [self.Tmat321[0][1]] + self.Tmat321[1],  # Tyx, Tyy, Tyz
            [self.Tmat321[0][2], self.Tmat321[1][1]] + self.Tmat321[2]  # Tzx, Tzy, Tzz
        ]
        self.Tmat6 = [t for ts in self.Tmat321 for t in ts]

        builder.dense(ti.ij, (self.n_obs, self.n_cells)).place(*self.Tmat6)

        builder.finalize()

    def build_kernel(self, model):
        kernel_matrix_forward(
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.base_cell_sizes, model,
            *self.Tmat6, mat=np.empty(0), kernel_dtype=self.kernel_dt,
            write_to_mat=False, compressed=False
        )

    def solve(self, magnetization):
        if self.is_cpu and psutil.virtual_memory().percent < 45:
            # TODO: 进一步考虑模型大小来选择求解方案，模型较大时也应该使用 `solve_Tx_b`
            return solve_Ax_b(merge_Tmat_as_A(self.Tmat33, direct_to_host=self.direct_to_host), magnetization)
        else:
            return solve_Tx_b(self.Tmat321, magnetization, progress=self.progress)


def solve_Tx_b(Tmat321, m, progress: bool = False):
    dtype = to_taichi_type(m.dtype)

    @ti_func
    def extract(r, c):
        return (
            Tmat321[0][0][r, c], Tmat321[0][1][r, c], Tmat321[0][2][r, c],
            Tmat321[1][0][r, c], Tmat321[1][1][r, c],
            Tmat321[2][0][r, c]
        )

    @ti_kernel
    def linear(x: ti.template(), Ax: ti.template()):
        n_obs = Ax.shape[0] // 3
        n_cells = x.shape[0] // 3

        for r in range(n_obs):
            row = r * 3

            bx: dtype = 0
            by: dtype = 0
            bz: dtype = 0

            for c in range(0, n_cells):
                col = c * 3

                txx, txy, txz, tyy, tyz, tzz = extract(r, c)
                mx, my, mz = x[col + 0], x[col + 1], x[col + 2]

                bx += txx * mx + txy * my + txz * mz
                by += txy * mx + tyy * my + tyz * mz
                bz += txz * mx + tyz * my + tzz * mz

            Ax[row + 0] = bx
            Ax[row + 1] = by
            Ax[row + 2] = bz

    return matrix_free.cg(ti.linalg.LinearOperator(linear), m, progress=progress).to_numpy()


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
