from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from discretize import TensorMesh
from numpy.typing import ArrayLike

from metalpy.scab.modelling.object import Object
from metalpy.utils.numpy import ensure_as_numpy_array_collection
from metalpy.utils.type import is_numeric_array, get_first_key, ensure_set_key


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

    Notes
    -----
    map_xxx_to_xxx系列函数均不保证输出的向量是否为原向量的视图，
    即输出的向量有可能与输入向量共享内存空间，也有可能不是，因此仅作只读用途使用。
    """

    def __init__(self,
                 mesh: TensorMesh,
                 models: dict[str, ArrayLike] | ArrayLike | None = None,
                 ind_active: ArrayLike = None,
                 **kwargs: ArrayLike):
        from metalpy.scab.modelling import Scene

        self._mesh = mesh

        default_key = None
        converted_models = ensure_as_numpy_array_collection(kwargs)

        if models is not None:
            if is_numeric_array(models):
                default_key = ensure_set_key(
                    converted_models,
                    Object.DEFAULT_KEY,
                    np.asarray(models)
                )
            else:
                converted_models.update(ensure_as_numpy_array_collection(models))

        if default_key is None:
            default_key = get_first_key(converted_models)

        if ind_active is not None:
            ind_active = np.asarray(ind_active)
            if ind_active.dtype == bool:
                assert ind_active.shape[0] == self.n_cells, \
                    f'`ind_active` defines activeness for all mesh cells,' \
                    f' which must have same size with mesh cells.' \
                    f' Got {ind_active.shape[0]}, expected {self.n_cells}.'
            else:
                assert np.issubdtype(ind_active.dtype, np.integer), \
                    f'`ind_active` must be array of `bool` or `integer`. Got `{ind_active.dtype}`.'
        else:
            if len(converted_models) > 0:
                for key, model in converted_models.items():
                    assert model.shape[0] == self.n_cells, \
                        f'`ModelledMesh`\'s constructor expects models to contain all mesh cells\'s values' \
                        f' in order to build `active_index`.' \
                        f' Avoid this by either specifying `active_index` manually or' \
                        f' converting models["{key}"] to match all the mesh cells.' \
                        f' Got {model.shape[0]}, expected {self.n_cells}.'
                ind_active = Scene.compute_active(converted_models.values())
            else:
                ind_active = np.ones(mesh.n_cells, dtype=bool)

        self._models = converted_models
        self._ind_active = ind_active
        self.default_key = default_key

    @property
    def model(self):
        """获取默认model（active形式）

        Notes
        -----
        默认活跃的数据为：
        1. models的第一个键
        2. 若models为空，则为kwargs的第一个键
        3. 若kwargs也为空，无选中的激活数据，则为active_cells
        """
        return self.get_active_model()

    @property
    def mesh(self):
        return self._mesh

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
        return self.mesh.n_cells

    @property
    def default_key(self):
        return self._default_key

    @default_key.setter
    def default_key(self, k):
        if k is not None:
            assert k in self._models, f'Cannot set `default_key` to `{k}` which does not exist.'
        self._default_key = k

    @property
    def has_default_model(self):
        return self.default_key is not None

    @property
    def has_only_default_model(self):
        return len(self) == 1 and self.has_default_model

    def set_active_model(self, key):
        self.default_key = key

    def clear_active_model(self):
        self.default_key = None

    def get_raw_model(self, key):
        """直接获取模型数组，不保证返回结果是有效模型或完整模型
        """
        return self._models[key]

    def keys(self):
        return self._models.keys()

    def get_active_model(self, key=None, **kwargs):
        if key is None:
            if self.has_default_model:
                key = self.default_key
            else:
                return np.ones(self.n_active_cells, dtype=bool)

        return self.map_to_active(self.get_raw_model(key), **kwargs)

    def is_active_model(self, model):
        return model.shape[0] == self.n_active_cells

    def is_complete_model(self, model):
        return model.shape[0] == self.n_cells

    def is_subset(self, mask):
        mask = self.check_complete_mask(mask)
        return np.all(self.active_cells | mask == self.active_cells)

    def get_complete_model(self, key=None, **kwargs):
        if key is None:
            if self.has_default_model:
                key = self.default_key
            else:
                return self.active_cells

        return self.map_to_complete(self.get_raw_model(key), **kwargs)

    def _unsupported_model_exception(self, model):
        return ValueError(f'Unsupported model size {model.shape},'
                          f' which is neither complete model ({self.n_cells},)'
                          f' nor active model ({self.n_active_cells},)')

    def map_to_complete(self, model: ArrayLike, dtype=None, fill_inactive=None, copy=True):
        """从有效网格模型或完整网格模型映射到完整网格模型

        Parameters
        ----------
        model
            待映射的模型
        dtype
            映射模型的值类型
        fill_inactive
            映射到完整网格时非活跃网格的填充值，若指定填充值不为None，则一定会发生拷贝
        copy
            若为`False`，则输入模型满足所有条件时不进行拷贝，否则保证返回值与输入不共享内存空间

        Returns
        -------
        ret
            在完整网格下的模型数组
        """
        from metalpy.scab.modelling import Scene

        if self.n_active_cells == self.n_cells:
            fill_inactive = None

        if fill_inactive is not None:
            copy = True
            ftype = type(fill_inactive)
            if dtype is None:
                dtype = ftype
            else:
                assert np.can_cast(ftype, dtype), f'Cannot cast fill value with type `{ftype}` to {dtype}.'

        model = np.asarray(model)
        if self.is_complete_model(model):
            ret = model
            if dtype is not None:
                ret = ret.astype(dtype, copy=copy)
                copy = False

            if copy:
                ret = ret.copy()

            if fill_inactive is not None:
                ret[~self.active_cells] = fill_inactive
        elif self.is_active_model(model):
            base_type = dtype or model.dtype
            if fill_inactive is None:
                inactive_val = Scene.get_inactive_value_for_type(base_type)
            else:
                inactive_val = fill_inactive
            ret = np.full(self.n_cells, inactive_val, dtype=base_type)
            ret[self.active_cells] = model
        else:
            raise self._unsupported_model_exception(model)

        return ret

    def map_to_active(self, model: ArrayLike, dtype=None, copy=True):
        model = np.asarray(model)
        if self.is_active_model(model):
            ret = model
        elif self.is_complete_model(model):
            ret = model[self.active_cells]
        else:
            raise self._unsupported_model_exception(model)

        if dtype is not None:
            ret = ret.astype(dtype, copy=copy)
            copy = False

        if copy:
            ret = ret.copy()

        return ret

    def check_complete_mask(self, mask: ArrayLike):
        mask = np.asarray(mask)
        assert mask.ndim == 1, 'Indices must 1D vector.'

        if mask.dtype == bool:
            assert mask.shape[0] == self.n_cells, \
                f'Invalid complete mask (expected {self.n_cells}, got {mask.shape[0]}).'
            ret = mask
        else:
            # assumed to be fancy indexing
            ret = np.full(self.n_cells, False, dtype=bool)
            ret[mask] = True

        return ret

    def check_active_mask(self, mask: ArrayLike):
        mask = np.asarray(mask)
        assert mask.ndim == 1, 'Indices must 1D vector.'

        if mask.dtype == bool:
            assert mask.shape[0] == self.n_active_cells, \
                f'Invalid active mask (expected {self.n_active_cells}, got {mask.shape[0]}).'
            ret = mask
        else:
            # assumed to be fancy indexing
            ret = np.full(self.n_active_cells, False, dtype=bool)
            ret[mask] = True

        return ret

    def is_active_mask(self, mask: ArrayLike):
        mask = np.asarray(mask)
        if mask.ndim != 1:
            return False
        if mask.dtype == bool:
            return self.n_active_cells == mask.shape[0]
        else:
            # assumed to be fancy indexing
            return mask.max() < self.n_active_cells

    def is_complete_mask(self, mask: ArrayLike):
        mask = np.asarray(mask)
        if mask.ndim != 1:
            return False
        if mask.dtype == bool:
            return self.n_cells == mask.shape[0]
        else:
            # assumed to be fancy indexing
            return mask.max() < self.n_cells

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
        return self.get_active_model(item)

    def __setitem__(self, item, model: ArrayLike):
        model = np.asarray(model)
        assert model.shape[0] in (self.n_active_cells, self.n_cells), \
            f'`ModelledMesh` only accepts models matching active cells or all mesh cells.' \
            f' Got model with size {model.shape[0]}, expected {self.n_active_cells} or {self.n_cells}.'
        self._models[item] = model

    def __contains__(self, item):
        return item in self._models
    
    def to_polydata(self,
                    scalars: str | Iterable[str] | ArrayLike | bool = True,
                    extra_models: dict[str, ArrayLike] | None = None,
                    fill_inactive=None,
                    **kwargs: ArrayLike):
        """导出模型网格为PyVista模型

        Parameters
        ----------
        scalars
            需要导出的模型名称。
            True代表全部导出，False代表不导出包含的模型。
            通过字符串或字符串列表指定需要导出的模型。
            或者指定为数组以导出该指定模型。
        extra_models
            额外需要绑定的模型
        kwargs
            额外需要绑定的模型，但是以kwargs风格传入
        fill_inactive
            用于填充非活跃网格的值，为None时默认使用模型对应类型的默认值（一般为0，参见Scene.INACTIVE_XXX）

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

        default_key_active_cells = 'ACTIVE'
        active_scalars = None

        models = {}
        if not is_numeric_array(scalars):
            if scalars is True:
                active_scalars = self.default_key
            elif isinstance(scalars, str):
                active_scalars = scalars
            scalars = self._check_scalars(scalars)

            for scalar in scalars:
                if active_scalars is None:
                    active_scalars = scalar
                models[scalar] = self._models[scalar]
        else:
            # assumed to be model array
            key_model = 'model'

            active_scalars = ensure_set_key(
                models,
                key_model,
                scalars
            )

        if extra_models is not None:
            models.update(extra_models)
            if active_scalars is None:
                active_scalars = get_first_key(extra_models)

        models.update(kwargs)
        if active_scalars is None:
            active_scalars = get_first_key(kwargs)

        key_active_cells = ensure_set_key(
            models,
            default_key_active_cells,
            self._ind_active  # 保证为complete_model
        )
        if key_active_cells != default_key_active_cells:
            warnings.warn(f'Conflicted key with default key `{default_key_active_cells}` detected,'
                          f' renaming it to `{key_active_cells}`.')

        for key in models:
            model = models[key]
            model = self.map_to_complete(model, fill_inactive=fill_inactive)
            models[key] = model

        if active_scalars is None:
            active_scalars = key_active_cells

        ret: pv.RectilinearGrid = Scene.mesh_to_polydata(self.mesh, models)
        ret.set_active_scalars(active_scalars)

        return ret

    def extract(self,
                indices: ArrayLike,
                scalars: str | Iterable[str] | bool = True,
                shallow=False) -> 'ModelledMesh':
        """通过网格mask提取子网格

        Parameters
        ----------
        indices
            有效网格坐标系下的mask（bool数组或下标数组）。
        scalars
            需要保留到新ModelledMesh实例的一个或多个模型的键名。
            若为True，则保留所有模型，若为False，则只提取网格不提取模型。
        shallow
            指示是否浅拷贝模型，否则在可能的时候会使用原模型的视图，某些情况可能导致意外的修改

        Returns
        -------
        ret
            包含新的有效网格掩码的模型网格

        Notes
        -----
        如果传入完整网格坐标系的mask，则只提取其和当前有效网格的交集，保证结果为当前有效网格的子集、
        """
        if self.is_active_mask(indices):
            indices = self.check_active_mask(indices)
            new_active_cells = self.map_to_complete(indices)
        else:
            new_active_cells = self.check_complete_mask(indices)
            new_active_cells &= self.active_cells
            indices = self.map_to_active(new_active_cells)

        scalars = self._check_scalars(scalars)
        models = {}

        for model_key in scalars:
            model = self.get_raw_model(model_key)
            if self.is_active_model(model):
                model = model[indices]
            else:
                if not shallow:
                    model = model.copy()
            models[model_key] = model

        ret = ModelledMesh(self.mesh, models=models, ind_active=new_active_cells)

        if self.default_key in ret:
            ret.set_active_model(self.default_key)

        return ret

    def reactivate(self,
                   indices,
                   scalars: str | Iterable[str] | bool = True,
                   shallow=False,
                   ) -> 'ModelledMesh':
        """给定新的有效网格掩码，重新定义模型网格

        Parameters
        ----------
        indices
            全局坐标系下的新有效网格掩码，用于重新生成网格
        scalars
            需要保留到新ModelledMesh实例的一个或多个模型的键名。
            若为True，则保留所有模型，若为False，则只提取网格不提取模型。
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
        new_active_cells = self.check_complete_mask(indices)

        is_subset = self.is_subset(new_active_cells)
        if is_subset:
            return self.extract(self.map_to_active(new_active_cells), scalars=scalars, shallow=shallow)
        else:
            ret = ModelledMesh(self.mesh, ind_active=new_active_cells)
            scalars = self._check_scalars(scalars)

            for model_key in scalars:
                model = self.get_raw_model(model_key)

                if not self.is_complete_model(model):
                    from metalpy.scab.modelling import Scene

                    warnings.warn(f'Reactivated mesh with model for active cells.'
                                  f' May lead to unexpected result in non-overlapping cells,'
                                  f' where inactive cells are filled with default value'
                                  f' `{Scene.get_inactive_value_for_type(model.dtype)}`')

                    model = self.map_to_complete(model)
                else:
                    if not shallow:
                        model = model.copy()

                ret[model_key] = model

            if self.default_key in ret:
                ret.set_active_model(self.default_key)

            return ret

    def rebind(
        self,
        model: str | ArrayLike,
        scalars: str | Iterable[str] | bool = False,
        shallow=False
    ) -> 'ModelledMesh':
        """基于将网格重新绑定到model模型上或model作为字符串指定的模型上

        Parameters
        ----------
        model
            需要重新绑定的新模型或model作为字符串指定的需要重新绑定的模型
        scalars
            需要一同提取到新ModelledMesh实例的模型名。
            若scalars为`False`，则新实例仅包含model指定的模型。
            若scalars为`True`，则新实例包含model指定的模型和当前实例的全部模型。
            若scalars为字符串或字符串集合，则新实例包含model指定的模型和scalars指定的模型。
        shallow
            指示是否浅拷贝模型，否则在可能的时候会使用原模型的视图，某些情况可能导致意外的修改

        Returns
        -------
        ret
            包含新的有效网格掩码的模型网格
        """
        from metalpy.scab.modelling import Scene

        scalars = set(self._check_scalars(scalars))

        if isinstance(model, str):
            if model not in scalars:
                scalars.add(model)
            ref = self.get_raw_model(model)
            mask = Scene.is_active(ref)
        else:
            mask = Scene.is_active(model)

        if self.is_active_model(mask):
            ret = self.extract(mask, scalars=scalars, shallow=shallow)
        else:
            ret = self.reactivate(mask, scalars=scalars, shallow=shallow)

        if not isinstance(model, str):
            model = ensure_set_key(ret, Object.DEFAULT_KEY, model)

        ret.set_active_model(model)

        return ret

    def _check_scalars(self, scalars) -> Iterable[str]:
        if scalars is True:
            return self.keys()
        elif scalars is False:
            return tuple()
        elif isinstance(scalars, str):
            return (scalars,)
        else:
            return scalars
