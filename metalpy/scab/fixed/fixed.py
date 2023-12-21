from __future__ import annotations

from SimPEG.simulation import BaseSimulation

from metalpy.mexin import Patch, Mixin
from metalpy.mexin.utils import TypeMap
from metalpy.scab.distributed.policies import Distributable


class FixedContext(Mixin):
    _implementations = TypeMap(allow_match_parent=True)

    def __init__(self, this, **kwargs):
        super().__init__(this)
        self.kwargs = kwargs

    def post_apply(self, this):
        impl = FixedContext._implementations.get(type(this))

        if impl is None:
            return

        this.mixins.add(impl, **self.kwargs)


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
        self.add_mixin(BaseSimulation, FixedContext, dpred=self.dpred)


def __implements(target):
    def decorator(func):
        FixedContext._implementations.map(target, func)
        return func
    return decorator


@__implements('SimPEG.potential_fields.magnetics.simulation.Simulation3DDifferential')
def _():
    # 给 Simulation3DDifferential 实现了一个更合理的 dpred 方法
    from .potential_fields.magnetics.simulation import FixedSimulation3DDifferentialMixin
    return FixedSimulation3DDifferentialMixin
