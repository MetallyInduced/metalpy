import abc

import numpy as np
import taichi as ti
from taichi.lang.util import to_taichi_type

from metalpy.scab.utils.misc import Field
from metalpy.utils.taichi import ti_cfg
from metalpy.utils.type import CachedProperty
from .demag_solver_context import DemagSolverContext


class DemagnetizationSolver(abc.ABC):
    def __init__(
            self,
            context: DemagSolverContext,
            use_complete_mesh: bool = False
    ):
        self.context = context
        self._model = None

        self._use_complete_mesh = use_complete_mesh

        if not self.use_complete_mesh:
            self.context.extract_active()

    @property
    def mesh(self):
        return self.context.mesh

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        DemagnetizationSolver.is_uniform.invalidate(self)

    @property
    def has_active_cells_mask(self):
        return self.context.has_active_cells_mask

    @property
    def active_cells_mask(self):
        return self.context.active_cells_mask

    @property
    def use_complete_mesh(self):
        if self._use_complete_mesh:
            # solver指定使用完整网格
            return True
        elif not self.has_active_cells_mask:
            # 未指定有效网格，视为使用完整网格
            return True
        elif np.all(self.active_cells_mask):
            # 如果传入的mask中指定所有网格均为有效网格，则亦是使用完整网格
            return True
        else:
            return False

    @property
    def shape_cells(self):
        """三个方向上网格数目
        """
        return np.asarray(self.context.shape_cells)

    @property
    def local_shape_cells(self):
        """截断距离影响范围内包含的三个方向上网格数目（曼哈顿距离下）
        """
        return np.asarray(self.context.local_shape_cells)

    @property
    def receiver_locations(self):
        return self.context.receiver_locations

    @property
    def xn(self):
        return self.context.xn

    @property
    def yn(self):
        return self.context.yn

    @property
    def zn(self):
        return self.context.zn

    @property
    def base_cell_sizes(self):
        return self.context.base_cell_sizes

    @property
    def min_cell_sizes(self):
        return self.context.min_cell_sizes

    @property
    def max_cell_sizes(self):
        return self.context.max_cell_sizes

    @property
    def source_field(self):
        return self.context.source_field

    @property
    def progress(self):
        return self.context.progress

    @property
    def kernel_dtype(self):
        if self.context.kernel_dtype is not None:
            return self.context.kernel_dtype
        else:
            return np.result_type(self.xn, self.receiver_locations)

    @property
    def kernel_dt(self):
        """kernel_dtype的taichi对应类型
        """
        return to_taichi_type(self.kernel_dtype)

    @property
    def cutoff(self):
        return self.context.cutoff

    @property
    def n_cells(self):
        return self.context.n_cells

    @property
    def n_active_cells(self):
        return self.context.n_active_cells

    @property
    def n_total_cells(self):
        return self.context.n_total_cells

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
        model = np.asarray(model)

        if source_field is None:
            source_field = self.source_field
            assert source_field is not None, ('`source_field` must be specified'
                                              ' either when initializing the solver(source_field=...)'
                                              ' or when calling dpred(source_field=...).')

        source_field = Field(source_field)
        if self.model is None or np.any(self.model != model):
            self.model = np.copy(model)
            self.build_kernel(model)

        H0 = source_field.unit_vector
        H0 = np.tile(H0[None, :], self.n_cells).ravel()

        if self.use_complete_mesh:
            model_ = np.full(self.n_cells, 0, dtype=model.dtype)
            model_[self.active_cells_mask] = model
            model = model_

        X = np.tile(model[:, None], 3).ravel()
        magnetization = X * H0

        ret = self.solve(magnetization, model=model)
        ret = ret.reshape(-1, 3)

        if self.use_complete_mesh:
            ret = ret[self.active_cells_mask]

        return ret

    @CachedProperty
    def is_uniform(self):
        """判断网格磁化率模型是否均匀，即所有网格磁化率相同
        """
        return np.allclose(self.model, self.model[0])

    @property
    def is_kernel_built_with_model(self):
        """如果为均匀网格，则核矩阵 A=I-KT 通常可以直接构建；
        否则需要在求解时构造 Ax = (I-KT)x = x - K(Tx)。
        """
        return self.is_uniform

    @abc.abstractmethod
    def build_kernel(self, model):
        """构造核矩阵
        """
        pass

    @abc.abstractmethod
    def solve(self, magnetization, model):
        """求解退磁效应下每个网格的三轴等效磁化率

        Parameters
        ----------
        magnetization
            磁化矩阵
        model
            磁化率模型

        Returns
        -------
        ret
            三轴等效磁化率矩阵
        """
        pass
