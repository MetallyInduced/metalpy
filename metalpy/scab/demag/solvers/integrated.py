import numpy as np

from metalpy.utils.taichi import ti_ndarray
from metalpy.utils.ti_solvers.solver_progress import ProgressList
from metalpy.utils.type import notify_package
from .demag_solver_context import DemagSolverContext
from .kernel import kernel_matrix_forward
from .solver import DemagnetizationSolver


class IntegratedSolver(DemagnetizationSolver):
    def __init__(
            self,
            context: DemagSolverContext
    ):
        """该函数直接计算核矩阵

        Parameters
        ----------
        context
            退磁求解上下文

        Notes
        -----
        优势：

        - 简单直观

        缺陷：

        - 存在taichi的int32索引限制
        """
        super().__init__(context)

        shape = (3 * self.n_obs, 3 * self.n_cells)
        if self.is_cpu:
            self.A = np.empty(shape, dtype=self.kernel_dtype)
        else:
            self.A = ti_ndarray(shape, dtype=self.kernel_dtype)

    def build_kernel(self, model):
        kernel_matrix_forward(
            self.receiver_locations,
            self.xn, self.yn, self.zn, model,
            mat=self.A,
            kernel_dtype=self.kernel_dt,
            apply_susc_model=True
        )

    def solve(self, magnetization, model):
        if self.is_cpu:
            Amat = self.A
        else:
            Amat = self.A.to_numpy()
        return solve_Ax_b(Amat, magnetization, progress=self.progress)

    @property
    def is_kernel_built_with_model(self):
        return True


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
