from __future__ import annotations

from SimPEG import maps
from SimPEG.simulation import LinearSimulation, BaseSimulation

from metalpy.scab.modelling.modelled_mesh import ModelledMesh
from metalpy.utils.type import undefined
from . import SimulationBuilder


class BaseSimulationBuilder(SimulationBuilder):
    def __init__(self, sim_cls: type[BaseSimulation]):
        super().__init__(sim_cls)
        self._modelled_mesh: ModelledMesh | None = None

    @property
    def has_mesh(self):
        return self._modelled_mesh is not None

    @property
    def modelled_mesh(self):
        assert self.has_mesh, 'Set mesh with `mesh(...)` or `active_mesh(...)` first.'
        return self._modelled_mesh

    @modelled_mesh.setter
    def modelled_mesh(self, m):
        self._modelled_mesh = m

    @property
    def n_active_cells(self):
        return self.modelled_mesh.n_active_cells

    @property
    def base_mesh(self):
        return self.modelled_mesh.mesh

    @property
    def ind_active(self):
        return self.modelled_mesh.active_cells

    @SimulationBuilder._supplies('mesh')
    def mesh(self, mesh):
        self.active_mesh(mesh)
        return undefined

    @SimulationBuilder._supplies('mesh')
    def active_mesh(self, mesh, ind_active=undefined):
        if isinstance(mesh, ModelledMesh):
            self.modelled_mesh = mesh
            mesh = mesh.mesh
        else:
            if undefined != ind_active:
                self.modelled_mesh = ModelledMesh(mesh, ind_active=ind_active)
            else:
                self.modelled_mesh = ModelledMesh(mesh)

        return mesh


class LinearSimulationBuilder(BaseSimulationBuilder):
    def __init__(self, sim_cls: type[LinearSimulation]):
        super().__init__(sim_cls)

    @SimulationBuilder._supplies('model_map')
    def model_map(self, mapping: maps.IdentityMap):
        return mapping
