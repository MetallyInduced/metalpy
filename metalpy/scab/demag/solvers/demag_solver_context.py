import numpy as np
from discretize.base import BaseTensorMesh

from metalpy.scab.utils.misc import Field
from metalpy.utils.type import CachedProperty


class DemagSolverContext:
    def __init__(
            self,
            mesh: BaseTensorMesh,
            active_cells_mask: np.ndarray,
            source_field: Field,
            kernel_dtype=None,
            cutoff=np.inf,
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
        cutoff
            截断距离，当网格间距离超出截断距离，则不考虑之间的自退磁效应
        progress
            是否输出求解进度条，默认为False不输出
        """
        self.mesh = mesh
        self.active_cells_mask = active_cells_mask

        # 算法相关属性
        self.source_field = source_field
        self.kernel_dtype = kernel_dtype
        self.cutoff = cutoff
        self.progress = progress

    def extract_active(self):
        """提取有效网格对应的 *cell_centers* 和 *h_gridded* 并设置对应成员
        """
        extracted = [
            DemagSolverContext.cell_centers,
            DemagSolverContext.h_gridded,
        ]

        for prop in extracted:
            prop.set(self, prop.__get__(self)[self.active_cells_mask])

        for key in vars(DemagSolverContext):
            prop = getattr(DemagSolverContext, key)

            if prop in extracted:
                continue

            if isinstance(prop, CachedProperty):
                try:
                    prop.invalidate(key)
                except AttributeError:
                    pass

    @CachedProperty
    def cell_centers(self):
        """每个网格的中心点坐标
        """
        return self.mesh.cell_centers

    @CachedProperty
    def h_gridded(self):
        """每个网格的三轴长度
        """
        return self.mesh.h_gridded

    @CachedProperty
    def bsw(self):
        """每个网格的 *西南地* 方向角点坐标
        """
        return self.cell_centers - self.h_gridded / 2.0

    @CachedProperty
    def tne(self):
        """每个网格的 *东北天* 方向角点坐标
        """
        return self.cell_centers + self.h_gridded / 2.0

    @property
    def is_symmetric(self):
        """判断是否为对称网格，即所有网格尺寸相同
        """
        return np.allclose(self.h_gridded, self.h_gridded[0])

    @property
    def shape_cells(self):
        """三个方向上网格数目
        """
        return self.mesh.shape_cells

    @property
    def local_shape_cells(self):
        """截断距离影响范围内包含的三个方向上网格数目（曼哈顿距离下）
        """
        # 局部网格在某个方向上要求最大网格中心间距为 n×d 时，需要包含 (n + 1)×d 个网格才可以实现
        if np.isinf(self.cutoff):
            local_shape_cells = self.shape_cells
        else:
            local_shape_cells = np.ceil(self.cutoff / self.base_cell_sizes).astype(int) + 1
            local_shape_cells = np.row_stack([local_shape_cells, self.shape_cells]).min(axis=0)

        return local_shape_cells

    @property
    def n_cells(self):
        """网格数
        """
        return self.h_gridded.shape[0]

    @property
    def n_active_cells(self):
        """有效网格数
        """
        return np.count_nonzero(self.active_cells_mask)

    @property
    def n_obs(self):
        """观测点数（通常等于总网格数）
        """
        return self.receiver_locations.shape[0]

    @property
    def receiver_locations(self):
        """返回退磁计算中的观测点坐标
        """
        return self.cell_centers

    @CachedProperty
    def xn(self):
        """每个网格在 *x* 方向上的边界坐标
        """
        return np.c_[self.bsw[:, 0], self.tne[:, 0]]

    @CachedProperty
    def yn(self):
        """每个网格在 *y* 方向上的边界坐标
        """
        return np.c_[self.bsw[:, 1], self.tne[:, 1]]

    @CachedProperty
    def zn(self):
        """每个网格在 *z* 方向上的边界坐标
        """
        return np.c_[self.bsw[:, 2], self.tne[:, 2]]

    @property
    def base_cell_sizes(self):
        """返回基础网格尺寸，即最小网格尺寸
        """
        return self.min_cell_sizes

    @property
    def min_cell_sizes(self):
        """返回最小网格尺寸
        """
        return np.min(self.h_gridded, axis=0)

    @property
    def max_cell_sizes(self):
        """返回最大网格尺寸
        """
        return np.max(self.h_gridded, axis=0)
