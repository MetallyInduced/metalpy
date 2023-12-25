import numpy as np

from metalpy.scab.utils.misc import Field
from metalpy.utils.taichi import ti_ndarray
from metalpy.utils.ti_solvers.solver_progress import ProgressList
from metalpy.utils.type import notify_package
from .kernel import kernel_matrix_forward
from .solver import DemagnetizationSolver


class IntegratedSolver(DemagnetizationSolver):
    def __init__(
            self,
            receiver_locations: np.ndarray,
            xn: np.ndarray,
            yn: np.ndarray,
            zn: np.ndarray,
            base_cell_sizes: np.ndarray,
            source_field: Field,
            kernel_dtype=None,
            progress=False
    ):
        """该函数直接计算核矩阵

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

        Notes
        -----
        优势：

        - 简单直观

        缺陷：

        - 存在taichi的int32索引限制
        """
        super().__init__(receiver_locations, xn, yn, zn, base_cell_sizes, source_field, kernel_dtype, progress)

        shape = (3 * self.n_obs, 3 * self.n_cells)
        if self.is_cpu:
            self.A = np.empty(shape, dtype=self.kernel_dtype)
        else:
            self.A = ti_ndarray(shape, dtype=self.kernel_dtype)

    def build_kernel(self, model):
        kernel_matrix_forward(
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.base_cell_sizes, model,
            *[None] * 6, mat=self.A, kernel_dtype=self.kernel_dt,
            write_to_mat=True, compressed=False
        )

    def solve(self, magnetization):
        if self.is_cpu:
            Amat = self.A
        else:
            Amat = self.A.to_numpy()
        return solve_Ax_b(Amat, magnetization, progress=self.progress)


def solve_Ax_b(A, m, progress: bool = False):
    tol = 1e-5

    try:
        import pyamg

        norm = np.linalg.norm(m)
        x, _ = pyamg.krylov.bicgstab(
            A, m, tol=tol,
            residuals=ProgressList(tol * norm) if progress else None
        )
    except ModuleNotFoundError:
        notify_package(
            pkg_name='pyamg',
            reason='`pyamg` not found, falling back to slower version `np.linalg.solve`.'
        )
        x = np.linalg.solve(A, m)

    return x
