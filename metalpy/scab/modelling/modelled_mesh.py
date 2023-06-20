from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from discretize import TensorMesh

from metalpy.scab.modelling.object import Object
from metalpy.utils.type import is_numeric_array


class ModelledMesh:
    """
    用于封装包含模型的网格。
    mesh是基础网格，active_cells是有效网格的选择器。
    其中总网格数为n_cells，active_cells选中的有效网格数为n_active_cells。

    model根据映射关系，区分为active_model和complete_model。
    其中:
    active_model长度为n_active_cells，每个元素直接对应一个有效网格的模型值。
    complete_model长度为n_cells，每个元素直接对应一个基础网格的模型值，其中无效网格的模型值为Scene.INACTIVE_{type}，
    type为模型的数据类型。
    """

    def __init__(self,
                 mesh: TensorMesh,
                 models: dict[str, np.ndarray] | None = None,
                 ind_active: np.ndarray = None,
                 **kwargs: np.ndarray):
        from metalpy.scab.modelling import Scene

        self.base_mesh = mesh

        if models is not None:
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
    def model(self):
        """获取默认model（active形式）
        """
        return self.get_active_model()

    @property
    def mesh(self):
        return self.base_mesh

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

    def is_active_model(self, model):
        return model.shape[0] == self.n_active_cells

    def is_complete_model(self, model):
        return model.shape[0] == self.n_cells

    def get_complete_model(self, key=None):
        if key is None:
            if self.has_default_model:
                key = self.default_key
            else:
                raise RuntimeError('Mesh contains various models, please specify one by name.')

        return self.map_to_complete(self._models[key])

    def _unsupported_model_exception(self, model):
        return ValueError(f'Unsupported model size {model.shape},'
                          f' which is neither complete model ({self.n_cells},)'
                          f' nor active model ({self.n_active_cells},)')

    def map_to_complete(self, model: np.ndarray):
        from metalpy.scab.modelling import Scene

        if self.is_complete_model(model):
            ret = model
        elif self.is_active_model(model):
            inactive_val = Scene.get_inactive_value_for_type(model.dtype)
            ret = np.full(self.n_cells, inactive_val, dtype=model.dtype)
            ret[self.active_cells] = model
        else:
            raise self._unsupported_model_exception(model)

        return ret

    def map_to_active(self, model: np.ndarray):
        if self.is_active_model(model):
            ret = model
        elif self.is_complete_model(model):
            ret = model[self.active_cells]
        else:
            raise self._unsupported_model_exception(model)

        return ret

    def map_mask_to_complete(self, mask: np.ndarray):
        from metalpy.scab.modelling import Scene

        if mask.dtype == bool:
            ret = self.map_to_complete(mask)
        else:
            # assumed to be fancy indexing
            inactive_val = Scene.get_inactive_value_for_type(bool)
            ret = np.full(self.n_cells, inactive_val, dtype=bool)
            ret[mask] = True

        return ret

    def map_mask_to_active(self, mask: np.ndarray):
        from metalpy.scab.modelling import Scene

        if mask.dtype == bool:
            ret = self.map_to_active(mask)
        else:
            # assumed to be fancy indexing
            inactive_val = Scene.get_inactive_value_for_type(bool)
            ret = np.full(self.n_active_cells, inactive_val, dtype=bool)
            ret[mask] = True

        return ret

    def get(self, key, default=None):
        return self._models.get(key, default)

    def active_models(self):
        for key in self:
            yield key, self.get_active_model(key)

    def complete_models(self):
        for key in self:
            yield key, self.get_complete_model(key)

    def __iter__(self):
        yield from self._models

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
                    scalars: str | Iterable[str] | np.ndarray | None = None,
                    extra_models: dict[str, np.ndarray] | None = None,
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

        models = {}
        if not is_numeric_array(scalars):
            if scalars is None:
                scalars = self._models.keys()
            elif isinstance(scalars, str):
                scalars = (scalars,)

            for scalar in scalars:
                models[scalar] = self._models[scalar]

        if extra_models is not None:
            models.update(extra_models)
        models.update(kwargs)

        key_active_cells = 'ACTIVE'
        active_scalars = key_active_cells

        if isinstance(scalars, np.ndarray):
            # assumed to be model array
            key_active_model = 'model'

            while key_active_model in models:
                warnings.warn(f'Keys of `models` conflicts with default scalars key {key_active_model}, renaming it.')
                key_active_cells = f'_{key_active_model}_'
            models = {key_active_model: scalars}
            active_scalars = key_active_model

        while key_active_cells in models:
            warnings.warn(f'Keys of `models` conflicts with default key {key_active_cells}, renaming it.')
            key_active_cells = f'_{key_active_cells}_'

        models[key_active_cells] = self._ind_active  # ensured as complete model

        for key in models:
            model = models[key]
            model = self.map_to_complete(model)
            models[key] = model

        ret: pv.RectilinearGrid = Scene.mesh_to_polydata(self.base_mesh, models)
        ret.set_active_scalars(active_scalars)

        return ret

    def extract(self, indices, mesh_only=False, shallow=False) -> 'ModelledMesh':
        """基于有效网格坐标系下的网格下标，提取局部模型网格

        Parameters
        ----------
        indices
            有效网格坐标系下的新有效网格掩码，用于提取网格
        mesh_only
            指示是否只提取网格，不提取模型
        shallow
            指示是否浅拷贝模型，否则在可能的时候会使用原模型的视图，某些情况可能导致意外的修改

        Returns
        -------
        ret
            包含新的有效网格掩码的模型网格
        """
        new_active_cells = self.map_mask_to_complete(indices)

        models = {}

        if not mesh_only:
            for model_key in self:
                model = self.get_active_model(model_key)[indices]
                if not shallow:
                    model = model.copy()
                models[model_key] = model

        return ModelledMesh(self.mesh, models=models, ind_active=new_active_cells)

    def reactivate(self, indices, mesh_only=False, shallow=False) -> 'ModelledMesh':
        """给定新的有效网格掩码，重新定义模型网格

        Parameters
        ----------
        indices
            全局坐标系下的新有效网格掩码，用于重新生成网格
        mesh_only
            指示是否只提取网格，不提取模型
        shallow
            指示是否浅拷贝模型，否则在可能的时候会使用原模型的视图，某些情况可能导致意外的修改

        Returns
        -------
        ret
            包含新的有效网格掩码的模型网格

        Notes
        -----
            由于目前无效网格值会被定义为0，所以新的mask和旧有mask交集之外的值会是0。
            如果shallow为真，则不保证新旧mask交集之外的值，因此在多级reactivate调用下，shallow模式可能会导致大量不可预测的结果
        """
        new_active_cells = self.map_mask_to_complete(indices)
        ret = ModelledMesh(self.mesh, ind_active=new_active_cells)

        if not mesh_only:
            for model_key in self:
                model = self.get_complete_model(model_key)
                if not model.flags.owndata:
                    warnings.warn('Reactivation of shallow ModelledMesh is likely to produce unexpected result,'
                                  ' please specify `shallow=False` in last `reactivate` or `extract` call.')
                if shallow:
                    ret[model_key] = model
                else:
                    ret[model_key] = model[new_active_cells].copy()

        return ret
