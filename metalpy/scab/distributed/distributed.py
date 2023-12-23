from __future__ import annotations

from SimPEG.simulation import BaseSimulation

from metalpy.mepa import Executor, ProcessExecutor
from metalpy.mexin.mixins import DispatcherMixin
from metalpy.mexin.patch import Patch


class _Distributed(DispatcherMixin, allow_match_parent=True):
    pass


class Distributed(Patch):
    Priority = Patch.DefaultPriority * 225  # 在Formatted之前

    def __init__(self, executor: Executor | int = None):
        super().__init__()
        if isinstance(executor, int):
            # 直接指定整数，默认启用进程池
            # TODO: 取消GIL后，可以考虑默认线程池？
            executor = ProcessExecutor(executor)
        self.executor = executor
        self.persisted_context = None

    def apply(self):
        self.add_mixin(
            BaseSimulation,
            _Distributed,
            patch=self
        )

    def unbind_context(self):
        """重写unbind_context来延长PatchContext的生命周期
        """
        self.persisted_context = self.context
        super().unbind_context()


@_Distributed.implements('SimPEG.potential_fields.base:BasePFSimulation')
def _():
    from .potential_fields.base import DistributedBasePFSimulationMixin
    return DistributedBasePFSimulationMixin
