import abc

import numpy as np

from metalpy.scab.utils.misc import Field


class DemagnetizationSolver(abc.ABC):
    def __init__(
            self,
            receiver_locations: np.ndarray,
            xn: np.ndarray,
            yn: np.ndarray,
            zn: np.ndarray,
            base_cell_sizes: np.ndarray,
            source_field: Field
    ):
        self.receiver_locations = receiver_locations
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.base_cell_sizes = base_cell_sizes
        self.source_field = source_field

        self.model = None

    @property
    def kernel_type(self):
        return np.result_type(self.xn, self.receiver_locations)

    def dpred(self, model):
        """计算退磁效应下每个网格的三轴等效磁化率

        Parameters
        ----------
        model
            磁化率模型

        Returns
        -------
        ret
            三轴等效磁化率矩阵
        """
        if self.model is None or np.any(self.model != model):
            self.build_kernel(model)
            self.model = model

        nC = self.xn.shape[0]
        H0 = self.source_field.unit_vector
        H0 = np.tile(H0[None, :], nC).ravel()
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
