from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from discretize import TensorMesh

from metalpy.scab.modelling.object import Object


class ModelledMesh:
    def __init__(self,
                 mesh: TensorMesh,
                 models: dict[str, np.ndarray] | None = None,
                 ind_active: np.ndarray = None,
                 **kwargs: np.ndarray):
        from metalpy.scab.modelling import Scene

        self.base_mesh = mesh

        kwargs.update(models)
        self._models = kwargs

        if ind_active is not None:
            if ind_active.dtype == bool:
                assert ind_active.shape[0] == self.n_cells, \
                    f'`ind_active` defines activeness for all mesh cells,' \
                    f' which must have same size with mesh cells.' \
                    f' Got {ind_active.shape[0]}, expected {self.n_cells}.'
            else:
                assert np.issubdtype(ind_active.dtype, np.integer), \
                    f'`ind_active` must be array of `bool` or `integer`. Got `{ind_active.dtype}`.'
        else:
            for key, model in kwargs.items():
                assert model.shape[0] == self.n_cells, \
                    f'`ModelledMesh`\'s constructor expects models to contain all mesh cells\'s values' \
                    f' in order to build `active_index`.' \
                    f' Avoid this by either specifying `active_index` manually or' \
                    f' converting models["{key}"] to match all the mesh cells.' \
                    f' Got {model.shape[0]}, expected {self.n_cells}.'
            ind_active = Scene.compute_active(kwargs.values())

        if ind_active is None:
            ind_active = np.ones(mesh.n_cells, dtype=bool)

        self._ind_active = ind_active
        self._default_key = None

    @property
    def active_cells(self):
        return self._ind_active

    @active_cells.setter
    def active_cells(self, val):
        self._ind_active = val

    @property
    def n_active_cells(self):
        if self._ind_active.dtype == bool:
            return np.count_nonzero(self._ind_active)
        else:
            return self._ind_active.shape[0]

    @property
    def n_cells(self):
        return self.base_mesh.n_cells

    @property
    def default_key(self):
        return self._default_key or Object.DEFAULT_KEY

    @property
    def has_default_model(self):
        return self.default_key in self

    @property
    def has_only_default_model(self):
        return len(self) == 1 and self.has_default_model

    def get_active_model(self, key=None):
        if key is None:
            if self.has_default_model:
                key = self.default_key
            else:
                raise RuntimeError('Mesh contains various models, please specify one by name.')

        return self.map_to_active(self._models[key])

    def get_complete_model(self, key=None):
        if key is None:
            if self.has_default_model:
                key = self.default_key
            else:
                raise RuntimeError('Mesh contains various models, please specify one by name.')

        return self.map_to_complete(self._models[key])

    def map_to_complete(self, model: np.ndarray):
        from metalpy.scab.modelling import Scene

        if model.shape[0] == self.n_cells:
            return model
        else:
            inactive_val = Scene.get_inactive_value_for_type(model.dtype)
            ret = np.repeat(np.array(inactive_val, dtype=model.dtype), self.n_cells)
            ret[self.active_cells] = model
            return ret

    def map_to_active(self, model: np.ndarray):
        if model.shape[0] == self.n_active_cells:
            return model
        else:
            return model[self.active_cells]

    def get(self, key, default=None):
        return self._models.get(key, default)

    def __len__(self):
        return len(self._models)

    def __getitem__(self, item):
        ret = self._models[item]
        if ret.shape[0] == self.n_active_cells:
            return ret
        else:
            return ret[self.active_cells]

    def __setitem__(self, item, model):
        assert model.shape[0] in (self.n_active_cells, self.n_cells), \
            f'`ModelledMesh` only accepts models matching active cells or all mesh cells.' \
            f' Got model with size {model.shape[0]}, expected {self.n_active_cells} or {self.n_cells}.'
        self._models[item] = model

    def __contains__(self, item):
        return item in self._models
    
    def to_polydata(self,
                    scalars: str | Iterable[str] | None = None,
                    extra_models: np.ndarray | dict[str, np.ndarray] | None = None,
                    **kwargs: np.ndarray):
        """导出模型网格为PyVista模型

        Parameters
        ----------
        scalars
            需要导出的模型名称，None代表全部导出
        extra_models
            额外需要绑定的模型
        kwargs
            额外需要绑定的模型，但是以kwargs风格传入

        Returns
        -------
        ret
            包含指定的需要导出的模型和额外模型的pv.PolyData实例

        Notes
        -----
        虽然名字为to_polydata，但他的返回值并不是pv.PolyData而是pv.RectilinearGrid
        """
        import pyvista as pv
        from metalpy.scab.modelling import Scene

        if extra_models is not None:
            if not isinstance(extra_models, dict):
                models = {'model': extra_models}
            else:
                models = extra_models.copy()
        else:
            models = {}

        if scalars is None:
            scalars = self._models.keys()
        elif isinstance(scalars, str):
            scalars = (scalars,)

        for scalar in scalars:
            models[scalar] = self._models[scalar]

        for key in models:
            models[key] = self.map_to_complete(models[key])

        active_key = 'ACTIVE'
        while active_key in models:
            warnings.warn(f'Keys of `models` conflicts with default key {active_key}, renaming it.')
            active_key = f'_{active_key}_'

        models[active_key] = self._ind_active

        ret: pv.RectilinearGrid = Scene.mesh_to_polydata(self.base_mesh, models)
        ret.set_active_scalars(active_key)

        return ret
