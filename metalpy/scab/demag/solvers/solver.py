import abc

import numpy as np
import taichi as ti
from taichi.lang.util import to_taichi_type

from metalpy.scab.utils.misc import Field
from metalpy.utils.taichi import ti_cfg


class DemagnetizationSolver(abc.ABC):
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
        self.receiver_locations = receiver_locations
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.base_cell_sizes = base_cell_sizes
        self.source_field = source_field

        self.model = None
        self._kernel_dtype = kernel_dtype

        self.progress = progress

    @property
    def kernel_dtype(self):
        if self._kernel_dtype is not None:
            return self._kernel_dtype
        else:
            return np.result_type(self.xn, self.receiver_locations)

    @property
    def kernel_dt(self):
        """kernel_dtype的taichi对应类型
        """
        return to_taichi_type(self.kernel_dtype)

    @property
    def n_cells(self):
        return self.xn.shape[0]

    @property
    def n_obs(self):
        return self.receiver_locations.shape[0]

    @property
    def n_cells_on_each_axes(self):
        """估算每个轴向上的网格数，采用网格坐标范围除以最小网格尺寸，获得网格数的上界
        """
        return [
            (self.xn.max() - self.xn.min()) / self.base_cell_sizes[0],
            (self.yn.max() - self.yn.min()) / self.base_cell_sizes[1],
            (self.zn.max() - self.zn.min()) / self.base_cell_sizes[2],
        ]

    @property
    def h_gridded(self):
        return (np.c_[self.xn[:, 1], self.yn[:, 1], self.zn[:, 1]] - self.receiver_locations) * 2

    @property
    def arch(self):
        return ti_cfg().arch

    @property
    def is_cpu(self):
        return self.arch == ti.cpu

    def dpred(self, model, source_field=None):
        """计算退磁效应下每个网格的三轴等效磁化率

        Parameters
        ----------
        model
            磁化率模型
        source_field
            外部场源，覆盖求解器定义时给定的场源信息。
            如果为None，则采用给定的默认场源信息

        Returns
        -------
        ret
            三轴等效磁化率矩阵
        """
        if source_field is None:
            source_field = self.source_field
            assert source_field is not None, ('`source_field` must be specified'
                                              ' either when initializing the solver(source_field=...)'
                                              ' or when calling dpred(source_field=...).')

        source_field = Field(source_field)
        if self.model is None or np.any(self.model != model):
            self.build_kernel(model)
            self.model = np.copy(model)

        H0 = source_field.unit_vector
        H0 = np.tile(H0[None, :], self.n_cells).ravel()
        X = np.tile(model, 3).ravel()
        magnetization = X * H0

        return self.solve(magnetization)

    @abc.abstractmethod
    def build_kernel(self, model):
        """构造核矩阵
        """
        pass

    @abc.abstractmethod
    def solve(self, magnetization):
        """求解退磁效应下每个网格的三轴等效磁化率

        Parameters
        ----------
        magnetization
            磁化矩阵

        Returns
        -------
        ret
            三轴等效磁化率矩阵
        """
        pass
