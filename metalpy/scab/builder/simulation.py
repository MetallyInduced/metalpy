from __future__ import annotations

from SimPEG import maps
from SimPEG.simulation import LinearSimulation

from . import SimulationBuilder


class LinearSimulationBuilder(SimulationBuilder):
    def __init__(self, sim_cls: LinearSimulation):
        super().__init__(sim_cls)

    @SimulationBuilder._supplies('model_map')
    def model_map(self, mapping: maps.IdentityMap):
        return mapping
