from __future__ import annotations

from SimPEG.simulation import BaseSimulation

from metalpy.mexin import Patch
from metalpy.mexin.mixins import DispatcherMixin
from metalpy.scab.distributed.policies import Distributable


class _Fixed(DispatcherMixin, allow_match_parent=True):
    pass


class Fixed(Patch, Distributable):
    def __init__(self, dpred=True):
        """一些潜在的修复？
        
        Parameters
        ----------
        dpred
            指示是否针对某些正演类添加更合理的 dpred 方法 （可能会随着SimPEG更新而过时）
        """
        super().__init__()
        self.dpred = dpred

    def apply(self):
        self.add_mixin(BaseSimulation, _Fixed, dpred=self.dpred)


@_Fixed.implements('SimPEG.potential_fields.magnetics.simulation:Simulation3DDifferential')
def _():
    # 给 Simulation3DDifferential 实现了一个更合理的 dpred 方法
    from .potential_fields.magnetics.simulation import FixedSimulation3DDifferentialMixin
    return FixedSimulation3DDifferentialMixin
