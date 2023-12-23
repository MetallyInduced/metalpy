from __future__ import annotations

from SimPEG.simulation import BaseSimulation

from metalpy.mexin import Patch
from metalpy.mexin.mixins import DispatcherMixin
from metalpy.scab.distributed.policies import NotDistributable


class _Formatted(DispatcherMixin, allow_match_parent=True):
    pass


class Formatted(Patch, NotDistributable):
    def __init__(self, pandas=False, locations=False):
        """格式化SimPEG正演类的预测结果，将结果从展平的状态转换为二维矩阵

        Parameters
        ----------
        pandas
            如果为True，则格式化为 `pandas.DataFrame` ，行名为数据名（例如 'tmi' ， 'bx' 等）
        locations
            如果为True，则在格式化结果的左侧附加观测点的坐标列
        """
        super().__init__()
        self.pandas = pandas
        self.locations = locations

    def apply(self):
        self.add_mixin(BaseSimulation, _Formatted, pandas=self.pandas, locations=self.locations)


@_Formatted.implements('SimPEG.potential_fields.base:BasePFSimulation')
def _():
    from .potential_fields.base import FormattedBasePFSimulationMixin
    return FormattedBasePFSimulationMixin


@_Formatted.implements('SimPEG.potential_fields.magnetics.simulation:Simulation3DDifferential')
def _():
    from .potential_fields.magnetics.simulation import FormattedSimulation3DDifferentialMixin
    return FormattedSimulation3DDifferentialMixin
