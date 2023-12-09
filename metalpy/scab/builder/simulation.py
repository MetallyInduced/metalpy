from __future__ import annotations

from SimPEG import maps
from SimPEG.simulation import LinearSimulation, BaseSimulation

from metalpy.scab.modelling.modelled_mesh import ModelledMesh
from metalpy.utils.type import undefined
from . import SimulationBuilder


class BaseSimulationBuilder(SimulationBuilder):
    def __init__(self, sim_cls: BaseSimulation):
        super().__init__(sim_cls)
        self._modelled_mesh: ModelledMesh | None = None

    @property
    def n_active_cells(self):
        assert self._modelled_mesh is not None, 'Set mesh with `mesh(...)` or `active_mesh(...)` first.'
        return self._modelled_mesh.n_active_cells

    @property
    def base_mesh(self):
        assert self._modelled_mesh is not None, 'Set mesh with `mesh(...)` or `active_mesh(...)` first.'
        return self._modelled_mesh.mesh

    @property
    def ind_active(self):
        assert self._modelled_mesh is not None, 'Set mesh with `mesh(...)` or `active_mesh(...)` first.'
        return self._modelled_mesh.active_cells

    def mesh(self, mesh):
        self.active_mesh(mesh)

    @SimulationBuilder._supplies('mesh')
    def active_mesh(self, mesh, ind_active=undefined):
        if isinstance(mesh, ModelledMesh):
            self._modelled_mesh = mesh
            mesh = mesh.mesh
        else:
            if undefined != ind_active:
                self._modelled_mesh = ModelledMesh(mesh, ind_active=ind_active)
            else:
                self._modelled_mesh = ModelledMesh(mesh)

        return mesh


class LinearSimulationBuilder(BaseSimulationBuilder):
    def __init__(self, sim_cls: LinearSimulation):
        super().__init__(sim_cls)

    @SimulationBuilder._supplies('model_map')
    def model_map(self, mapping: maps.IdentityMap):
        return mapping
