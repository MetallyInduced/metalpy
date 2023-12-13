from __future__ import annotations

import os

from SimPEG.potential_fields.base import BasePFSimulation

from metalpy.utils.type import undefined
from .. import SimulationBuilder
from ..simulation import LinearSimulationBuilder


class BasePFSimulationBuilder(LinearSimulationBuilder):
    def __init__(self, sim_cls: type[BasePFSimulation]):
        super().__init__(sim_cls)

    @SimulationBuilder._supplies(['ind_active', 'actInd'])
    def active_mesh(self, mesh, ind_active=undefined):
        super().active_mesh(mesh, ind_active)
        return self.ind_active

    @SimulationBuilder._supplies('store_sensitivities', 'sensitivity_path')
    def store_sensitivities(self, target_or_path: bool | str):
        if target_or_path is True:
            return 'ram', undefined
        elif target_or_path is False:
            return 'forward_only', undefined
        else:
            return 'disk', os.fspath(target_or_path)

    @SimulationBuilder._supplies('sensitivity_dtype')
    def sensitivity_dtype(self, dtype):
        """指定灵敏度矩阵的数据类型

        Parameters
        ----------
        dtype
            灵敏度矩阵类型

        Notes
        -----
        注意：在 `forward_only` 模式下，不会存储灵敏度矩阵，此时计算结果强制使用 `float64`
        """
        return dtype

    @SimulationBuilder._supplies('n_processes')
    def n_processes(self, n: int | None):
        """指定并行进程数，
        为1时代表串行执行（默认选项），
        为None时使用全部可用逻辑核心数，
        为负数时为全部可用逻辑核心数 - n（即指定闲置核心数）

        Parameters
        ----------
        n
            并行进程数
        """
        if n < 0:
            n = os.cpu_count() + n

        return n
