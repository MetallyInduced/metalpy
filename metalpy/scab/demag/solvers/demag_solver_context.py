import numpy as np
from discretize.base import BaseTensorMesh

from metalpy.scab.utils.misc import Field


class DemagSolverContext:
    def __init__(
            self,
            mesh: BaseTensorMesh,
            active_cells_mask: np.ndarray,
            source_field: Field,
            kernel_dtype=None,
            progress: bool = False
    ):
        """该函数将矩阵分为9个部分，从而实现一些优化

        Parameters
        ----------
        mesh
            网格
        active_cells_mask
            有效网格掩码
        source_field
            定义默认外部场源，求解时若未指定场源，则使用该值
        kernel_dtype
            核矩阵数据类型，默认为None，自动从输入数据推断
        progress
            是否输出求解进度条，默认为False不输出
        """
        self.mesh = mesh
        self.active_cells_mask = active_cells_mask

        # 网格相关属性
        self.cell_centers = mesh.cell_centers
        self.h_gridded = mesh.h_gridded
        self.bsw = self.cell_centers - self.h_gridded / 2.0
        self.tne = self.cell_centers + self.h_gridded / 2.0

        # 算法相关属性
        self.source_field = source_field
        self.kernel_dtype = kernel_dtype
        self.progress = progress

    def extract_active(self):
        self.cell_centers = self.cell_centers[self.active_cells_mask]
        self.h_gridded = self.h_gridded[self.active_cells_mask]
        self.bsw = self.bsw[self.active_cells_mask]
        self.tne = self.tne[self.active_cells_mask]

    @property
    def is_symmetric(self):
        """判断是否为对称网格，即所有网格尺寸相同
        """
        return np.allclose(self.h_gridded, self.h_gridded[0])

    @property
    def shape_cells(self):
        return self.mesh.shape_cells

    @property
    def n_cells(self):
        return self.h_gridded.shape[0]

    @property
    def n_active_cells(self):
        return np.count_nonzero(self.active_cells_mask)

    @property
    def n_obs(self):
        return self.receiver_locations.shape[0]

    @property
    def receiver_locations(self):
        """返回退磁计算中的观测点坐标
        """
        return self.cell_centers

    @property
    def xn(self):
        return np.c_[self.bsw[:, 0], self.tne[:, 0]]

    @property
    def yn(self):
        return np.c_[self.bsw[:, 1], self.tne[:, 1]]

    @property
    def zn(self):
        return np.c_[self.bsw[:, 2], self.tne[:, 2]]

    @property
    def base_cell_sizes(self):
        """返回最小网格尺寸
        """
        return np.min(self.h_gridded, axis=0)
