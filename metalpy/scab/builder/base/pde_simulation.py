from __future__ import annotations

import numpy as np
from SimPEG import base, maps
from SimPEG.maps import ComboMap

from metalpy.utils.type import undefined
from .. import SimulationBuilder
from ..simulation import BaseSimulationBuilder


class BasePDESimulationBuilder(BaseSimulationBuilder):
    def __init__(self, sim_cls: base.BasePDESimulation):
        super().__init__(sim_cls)


class BaseMagneticPDESimulationBuilder(BasePDESimulationBuilder):
    def __init__(self, sim_cls: base.BaseMagneticPDESimulation):
        super().__init__(sim_cls)
        self._mu_map = None
        self._chi_map = None

    def chi_map(self, chi_map: maps.IdentityMap | None = None):
        """指定将磁化率模型映射到磁导率模型

        Parameters
        ----------
        chi_map
            磁化率模型的映射（ ChiMap * chi_map * model ），
            若为None则只将磁化率映射为磁导率（ ChiMap * model ）
        """
        if chi_map is not None:
            self._chi_map = [maps.ChiMap(), chi_map]
        else:
            self._chi_map = [maps.ChiMap()]

    @SimulationBuilder._supplies('mu')
    def mu(self, mu=undefined, map=undefined):
        if undefined == map:
            self._mu_map = map
        return mu

    @SimulationBuilder._supplies('mui', 'muiMap')
    def mui(self, mui=undefined, map=undefined):
        return mui, map

    @SimulationBuilder._assembles('muMap')
    def _mu_mapping(self):
        inactive_values = 1
        mappings = []

        if self._mu_map is not None:
            mappings.append(self._mu_map)

        if self._chi_map is not None:
            inactive_values = 0
            mappings.extend(self._chi_map)

        if not np.all(self.ind_active):
            mappings.append(maps.InjectActiveCells(
                self.base_mesh,
                self.ind_active,
                valInactive=inactive_values
            ))

        if len(mappings) == 0:
            return maps.IdentityMap(nP=self.n_active_cells)
        elif len(mappings) == 1:
            return mappings[0]
        else:
            return ComboMap(mappings)
