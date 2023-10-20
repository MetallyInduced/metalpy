import numpy as np

from metalpy.scab.utils.misc import Field
from metalpy.utils.taichi import ti_FieldsBuilder
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
            is_cpu: bool
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
        is_cpu
            是否是CPU架构，否则需要在对应设备上分配返回值矩阵

        Notes
        -----
            优势：
            1. 简单直观

            缺陷：
            1. 存在taichi的int32索引限制
        """
        super().__init__(receiver_locations, xn, yn, zn, base_cell_sizes, source_field)
        self.is_cpu = is_cpu

        nC = xn.shape[0]
        nObs = receiver_locations.shape[0]

        if is_cpu:
            self.builder = None
            self.A = np.empty((3 * nObs, 3 * nC), dtype=self.kernel_type)
        else:
            self.builder = builder = ti_FieldsBuilder()
            self.A = builder.place_dense((3 * nObs, 3 * nC), self.kernel_type)

    def build_kernel(self, model):
        # TODO: 实现进度条
        kernel_matrix_forward(
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.base_cell_sizes, model,
            *[0] * 9, self.A,
            write_to_mat=True, compressed=False
        )

    def solve(self, magnetization):
        if self.is_cpu:
            Amat = self.A
        else:
            Amat = self.A.to_numpy()
        return solve_Ax_b(Amat, magnetization)


def solve_Ax_b(A, m):
    try:
        import pyamg
        x, _ = pyamg.krylov.bicgstab(A, m)
    except ModuleNotFoundError:
        notify_package(
            pkg_name='pyamg',
            reason='`pyamg` not found, falling back to slower version `np.linalg.solve`.'
        )
        x = np.linalg.solve(A, m)

    return x
