import numpy as np
import taichi as ti

from metalpy.utils.taichi import ti_ndarray
from .kernel import kernel_matrix_forward


def forward_integrated(
        magnetization: np.ndarray,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        base_cell_sizes: np.ndarray,
        susc_model: np.ndarray,
        is_cpu: bool
):
    """该函数直接计算核矩阵

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
    is_cpu
        是否是CPU架构，否则需要在对应设备上分配返回值矩阵

    Notes
    -----
        优势：
        1. 简单直观

        缺陷：
        1. 存在taichi的int32索引限制
    """
    nC = xn.shape[0]
    nObs = receiver_locations.shape[0]

    if is_cpu:
        A = np.empty((3 * nObs, 3 * nC), dtype=np.float64)
    else:
        A = ti_ndarray(dtype=ti.f64, shape=(3 * nObs, 3 * nC))

    kernel_matrix_forward(receiver_locations, xn, yn, zn, base_cell_sizes, susc_model,
                          *[0] * 9, A, write_to_mat=True, compressed=False)

    if is_cpu:
        Amat = A
    else:
        Amat = A.to_numpy()

    return solve_Ax_b(Amat, magnetization)


def solve_Ax_b(A, m):
    import pyamg
    x, _ = pyamg.krylov.bicgstab(A, m)
    return x
