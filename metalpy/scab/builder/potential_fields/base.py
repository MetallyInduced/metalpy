from __future__ import annotations

import os

from SimPEG.potential_fields.base import BasePFSimulation

from metalpy.utils.type import undefined
from .. import SimulationBuilder
from ..simulation import LinearSimulationBuilder


class BasePFSimulationBuilder(LinearSimulationBuilder):
    def __init__(self, sim_cls: BasePFSimulation):
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
