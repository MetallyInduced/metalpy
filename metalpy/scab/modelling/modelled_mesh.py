from __future__ import annotations

import warnings
from typing import Iterable, TYPE_CHECKING, Literal, overload, Sequence

import numpy as np
from discretize import TensorMesh
from numpy.typing import ArrayLike

from metalpy.scab.modelling.formats.modelled_mesh import MeshZFormat
from metalpy.scab.modelling.object import Object
from metalpy.scab.modelling.shapes import Shape3D
from metalpy.utils.bounds import Bounds
from metalpy.utils.numpy import ensure_as_numpy_array_collection
from metalpy.utils.type import is_numeric_array, get_first_key, ensure_set_key, Self

if TYPE_CHECKING:
    import pyvista as pv


class ModelledMesh(MeshZFormat):
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

    `ind_active` 支持通过有效网格掩码或者有效网格索引来指定，在稀疏网格中后者性能往往更好

    ModelledMesh现在显式区分 `active_mask` 和 `active_indices` ：

    * `active_mask` 保证返回值为可计算的有效网格掩码；

    * `active_indices` 可以规避多余的网格掩码转换，但是只保证返回值可用于 `np.ndarray` 的索引操作，
      返回值可能为切片（“:”）、有效掩码（[True, False, True,...]）或有效元素索引（[0, 2, ...]）
      （详见[numpy索引格式](https://numpy.org/doc/stable/user/basics.indexing.html)）。
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
            self.active_cells = ind_active
        elif len(converted_models) > 0:
            for key, model in converted_models.items():
                assert model.shape[0] == self.n_cells, \
                    f'`ModelledMesh`\'s constructor expects models to contain all mesh cells\'s values' \
                    f' in order to build `active_index`.' \
                    f' Avoid this by either specifying `ind_active` manually or' \
                    f' converting models["{key}"] to match all the mesh cells.' \
                    f' Got {model.shape[0]}, expected {self.n_cells}.'
            ind_active = Scene.compute_active(converted_models.values())
            self.active_mask = ind_active
        else:
            self.active_cells = True

        self._models = converted_models
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
        """从兼容性考量，返回有效元素掩码，等价于 `active_mask`
        """
        return self.active_mask

    @active_cells.setter
    def active_cells(self, val):
        val = np.asarray(val)
        if val.dtype == bool:
            self.active_mask = val
        else:
            self.active_indices = val

    @property
    def active_mask(self):
        """返回有效元素掩码，如果存储的有效元素为索引形式，则可能导致额外的内存消耗和计算量

        如果只是直接用于从 complete model 中提取有效元素，可以考虑使用 `active_indices` 以获得更高的效率
        """
        if self.is_all_active or self.is_all_inactive:
            return np.full(self.n_cells, self._ind_active)
        elif self.is_active_indices:
            # 整型指定的稀疏索引格式，转换为掩码形式
            mask = np.zeros(self.n_cells, dtype=bool)
            mask[self._ind_active] = True
            return mask
        else:  # active mask形式
            return self._ind_active

    @active_mask.setter
    def active_mask(self, val):
        val = np.asarray(val)

        if np.ndim(val) == 0:
            # 如果val是0维标量，则取消ndarray的包装
            val = val.item()

        if val is True or val is False:
            self._ind_active = val
        else:
            assert val.shape[0] == self.n_cells, \
                f'`ind_active` defines activeness for all mesh cells,' \
                f' which must have same size with mesh cells.' \
                f' Got {val.shape[0]}, expected {self.n_cells}.'

            self._ind_active = val

    @property
    def active_indices(self):
        """返回有效元素索引，可以作为numpy array的索引，返回值可能为任何numpy支持的索引方式
        """
        if self.is_all_active:
            return slice(None)
        elif self.is_all_inactive:
            return []
        elif self.is_active_indices:
            return self._ind_active
        else:  # active mask形式
            return self.active_mask

    @active_indices.setter
    def active_indices(self, val):
        val = np.asarray(val)
        assert val.dtype == bool or np.issubdtype(val.dtype, np.integer), \
            f'`ind_active` must be array of `bool` or `integer`. Got `{val.dtype}`.'

        self._ind_active = val

    @property
    def is_active_indices(self):
        """指示有效网格是否是由稀疏下标指定（而非有效掩码）
        例如针对四个网格
        使用 [0, 2] 作为稀疏下标等价于 [True, False, True, False] 的有效掩码
        """
        return np.issubdtype(self._ind_active.dtype, np.integer)

    @property
    def is_all_active(self):
        return self._ind_active is True

    @property
    def is_all_inactive(self):
        return self._ind_active is False

    @property
    def n_active_cells(self):
        if self.is_all_active:
            return self.n_cells
        elif self.is_all_inactive:
            return 0
        elif self.is_active_indices:
            return len(self.active_indices)
        else:  # active mask形式
            return np.count_nonzero(self._ind_active)

    @property
    def n_cells(self):
        return self.mesh.n_cells

    @property
    def default_key(self) -> str:
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

    @property
    def cell_centers(self):
        return self.mesh.cell_centers

    @property
    def ndim(self):
        return self.mesh.dim

    @property
    def dim(self):
        return self.ndim

    @property
    def h(self):
        return self.mesh.h

    @property
    def active_cell_centers(self):
        return self.cell_centers[self.active_indices]

    @property
    def bounds(self):
        bounds = Bounds.unbounded(self.mesh.dim)
        dim_label = ['x', 'y', 'z']
        for dim in range(self.mesh.dim):
            n_vector = getattr(self.mesh, "nodes_" + dim_label[dim])
            bounds.set(dim, n_vector[0], n_vector[-1])

        return bounds

    @property
    def origin(self):
        return self.mesh.origin

    def set_active_model(self, key):
        self.default_key = key

    def clear_active_model(self):
        self.default_key = None

    def get_raw_model(self, key=None):
        """直接获取模型数组，不保证返回结果是有效模型或完整模型
        """
        if key is None:
            if self.has_default_model:
                key = self.default_key
            else:
                raise RuntimeError('No default model specified.')

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
        active_mask = self.active_mask
        return np.all(active_mask | mask == active_mask)

    def get_complete_model(self, key=None, **kwargs):
        if key is None:
            if self.has_default_model:
                key = self.default_key
            else:
                return self.active_mask

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
                ret[~self.active_mask] = fill_inactive
        elif self.is_active_model(model):
            base_type = dtype or model.dtype
            if fill_inactive is None:
                inactive_val = Scene.get_inactive_value_for_type(base_type)
            else:
                inactive_val = fill_inactive
            ret = np.full(self.n_cells, inactive_val, dtype=base_type)
            ret[self.active_indices] = model
        else:
            raise self._unsupported_model_exception(model)

        return ret

    def map_to_active(self, model: ArrayLike, dtype=None, copy=True):
        model = np.asarray(model)
        if self.is_active_model(model):
            ret = model
        elif self.is_complete_model(model):
            ret = model[self.active_indices]
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
            f' Got model {item} with size {model.shape[0]}, expected {self.n_active_cells} or {self.n_cells}.'
        self._models[item] = model

    def __delitem__(self, item):
        del self._models[item]
        if self.default_key == item:
            self.default_key = None
        if len(self) > 0:
            self.default_key = get_first_key(self)

    def __contains__(self, item):
        return item in self._models

    def plot(self,
             *,
             scalars: str | Iterable[str] | ArrayLike | bool = True,
             prune: Literal[True, False] = True,
             **kwargs
             ):
        if scalars is True:
            active_scalars = self.default_key
            if active_scalars is not None:
                scalars = active_scalars

        poly = self.to_polydata(scalars=scalars, prune=prune)

        poly.plot(**kwargs)

    @overload
    def to_polydata(self,
                    scalars: str | Iterable[str] | ArrayLike | bool = True,
                    extra_models: dict[str, ArrayLike] | None = None,
                    fill_inactive=None,
                    *,
                    prune: Literal[False] = False,
                    **kwargs: ArrayLike) -> 'pv.RectilinearGrid':
        ...

    @overload
    def to_polydata(self,
                    scalars: str | Iterable[str] | ArrayLike | bool = True,
                    extra_models: dict[str, ArrayLike] | None = None,
                    fill_inactive=None,
                    *,
                    prune: Literal[True],
                    **kwargs: ArrayLike) -> 'pv.UnstructuredGrid':
        ...

    def to_polydata(self,
                    scalars: str | Iterable[str] | ArrayLike | bool = True,
                    extra_models: dict[str, ArrayLike] | None = None,
                    fill_inactive=None,
                    prune: bool = True,
                    **kwargs: ArrayLike) -> 'pv.RectilinearGrid' | 'pv.UnstructuredGrid':
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
        prune
            从导出的 `RectilinearGrid` 中筛选活跃网格（会导致返回类型变为 `UnstructuredGrid` ）

        Returns
        -------
        ret
            包含指定的需要导出的模型和额外模型的pv.PolyData实例

        Notes
        -----
        虽然名字为to_polydata，但他的返回值并不是 `pv.PolyData` 而是 `pv.RectilinearGrid` 或 `pv.UnstructuredGrid`
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
            self.active_mask  # 保证为complete_model
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

        if prune and not self.is_all_active:
            return ret.extract_cells(self.active_indices)

        return ret

    def extract(self,
                indices_or_shape: ArrayLike | str | Shape3D,
                scalars: str | Iterable[str] | bool = True,
                shallow=False) -> Self:
        """通过网格mask提取子网格

        Parameters
        ----------
        indices_or_shape
            需要提取的子网格掩码（bool数组、下标数组或数据名字符串）。
            如果传入完整网格坐标系的mask，则只提取其和当前有效网格的交集，保证结果为当前有效网格的子集。
            如果传入Shape3D实例，则提取与该Shape3D相交的子网格。
        scalars
            需要保留到新ModelledMesh实例的一个或多个模型的键名。
            若为True，则保留所有模型，若为False，则只提取网格不提取模型。
        shallow
            指示是否浅拷贝模型，否则在可能的时候会使用原模型的视图，某些情况可能导致意外的修改

        Returns
        -------
        ret
            包含新的有效网格掩码的模型网格
        """
        from metalpy.scab.modelling import Scene

        if isinstance(indices_or_shape, str):
            indices = Scene.compute_active(self.get_active_model(indices_or_shape))
        elif isinstance(indices_or_shape, Shape3D):
            indices = indices_or_shape.place(self.active_cell_centers)
        else:
            indices = indices_or_shape
            if not self.is_active_mask(indices):
                new_active_cells = self.check_complete_mask(indices) & self.active_indices
                indices = self.map_to_active(new_active_cells)

        indices = self.check_active_mask(indices)
        new_active_cells = self.map_to_complete(indices)

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
                   ) -> Self:
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
    ) -> Self:
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
            if self.is_active_model(mask):
                # mask是self下的有效网格模型，意味着其不是ret下的有效网格模型
                # 因此model不能直接继承给ret，而需要重新映射为ret下的有效网格模型
                # 例如：假设self总网格3000个，有效网格1000个，ret含有效网格500个，
                # model作为self的有效网格模型，长度为1000，
                # 而ret只接受500或3000两种模型格式，因此需要重新映射到长度为500的格式
                model = model[mask]
            model = ensure_set_key(ret, Object.DEFAULT_KEY, model)

        ret.set_active_model(model)

        return ret

    def expand(
            self,
            new_bounds: ArrayLike | None = None,
            *,
            n_cells: ArrayLike | None = None,
            ratio: ArrayLike | None = None,
            precise: bool | None = None,
            proportion=None,
            increment=None
    ) -> Self:
        """扩张网格边界，边界网格支持指数增大网格或等距网格（指定ratio = 1）

        指数增长模式下，基础网格尺寸为对应方向边界网格的尺寸

        Parameters
        ----------
        new_bounds
            新的网格边界，支持通过 `Bounds.bounded` 指定部分边界
        n_cells
            增长的边界区域中各个方向上的网格数，np.nan代表不指定，可以使用 `Bounds.bounded` 来部分指定
        ratio
            增长的边界区域中各个方向上的增长倍率，np.nan代表不指定，可以使用 `Bounds.bounded` 来部分指定。
            可能会导致结果和 `new_bounds` 不严格重合，会超出期望的边界
        precise
            要求新的网格边界与 `new_bounds` 严格重合（收缩时会裁剪边缘网格，扩张时要求使用n_cells指定边界）。
            如果指定了 `ratio` ，则默认为 `False` ，否则默认为 `True`
        proportion, increment
            扩张既有网格边界或新指定边界的比例或增量，参见 `Bounds.expand` 的参数说明

        Returns
        -------
        ret
            扩张边界后的网格

        Notes
        -----
        `ratio` 和 `n_cells` 均未指定时，默认采用1.5的网格增大因子

        `ratio` 和 `n_cells` 均指定时，默认通过 `cells` 计算

        特殊地，如果在指定 `n_cells` 同时指定 `ratio=1` ，则采用等距网格进行扩张，否则在通过 `n_cells` 计算时无视 `ratio` 参数。
        """
        assert isinstance(self.mesh, TensorMesh), (
            f'`{ModelledMesh.expand.__name__}` supports only `TensorMesh` for now.'
        )

        old_bounds = self.bounds

        if new_bounds is not None:
            new_bounds = old_bounds.override(by=new_bounds)
        else:
            new_bounds = old_bounds

        if proportion is not None or increment is not None:
            new_bounds.expand(proportion=proportion, increment=increment, inplace=True)

        bounds_delta = new_bounds - self.bounds
        ndim = self.ndim

        if ratio is not None:
            if np.ndim(ratio) == 0:
                ratio = [ratio] * 6
            ratio = abs(Bounds(ratio))
            if precise is None:
                # 指定ratio，默认precise为False（ratio基本不可能实现精准对齐新网格边界）
                precise = False
        else:
            ratio = Bounds.unbounded(ndim)
            if precise is None:
                # 没有指定ratio，默认precise为True
                precise = True

        if n_cells is not None:
            if np.ndim(n_cells) == 0:
                n_cells = [n_cells] * 6
            n_cells = abs(Bounds(n_cells))
        else:
            n_cells = Bounds.unbounded(ndim)

        # 均未指定时，默认扩张区域的网格数恒定为5
        n_cells[np.isnan(n_cells) & np.isnan(ratio)] = 5

        old_h = [np.copy(h) for h in self.mesh.h]
        cells = []
        for axis in range(ndim):
            base_cell_size = self.mesh.h[axis][[0, -1]]
            delta = bounds_delta.get(axis)
            a_exps = ratio.get(axis)
            a_cells = n_cells.get(axis)

            old_h_axis = old_h[axis]
            current_h = [[], old_h_axis, []]

            if delta[0] < 0:
                specs, extra = compute_exponential_cell_specs(
                    -delta[0], base_cell_size[0], a_exps[0], a_cells[0],
                    precise=precise
                )
                current_h[0] = specs[::-1]
            else:
                h, extra = compute_trimmed_length(delta[0], current_h[1], precise)
                current_h[0] = DeleteAndExtend(len(current_h[1]) - len(h) + 1, arr=[h[0]])
                current_h[1] = h[1:]

            if delta[1] > 0:
                specs, extra = compute_exponential_cell_specs(
                    delta[1], base_cell_size[1], a_exps[1], a_cells[1],
                    precise=precise
                )
                current_h[2] = specs
            else:
                h, extra = compute_trimmed_length(-delta[1], current_h[1][::-1], precise)
                current_h[2] = DeleteAndExtend(len(current_h[1]) - len(h) + 1, arr=[h[0]])
                current_h[1] = h[:0:-1]

            cells.extend(current_h[::2])

        return self.expand_cells(cells=cells)

    def expand_cells(
            self,
            n_cells: ArrayLike | int = 0,
            ratio: ArrayLike | float = 1.0,
            cells: list[ArrayLike | DeleteAndExtend] | None = None,
    ) -> Self:
        """扩张网格边界，通过直接指定新网格的参数来扩张

        通过组合使用 `n_cells` 和 `ratio` 来自动计算新网格的参数

        通过直接指定 `cells` 来直接添加新的网格

        Parameters
        ----------
        n_cells
            各个方向上增长的网格数，np.nan代表不指定，可以使用 `Bounds.bounded` 来部分指定
        ratio
            各个方向上增长网格的增长倍率，np.nan代表不指定，可以使用 `Bounds.bounded` 来部分指定
        cells
            各个方向上增长网格的网格尺寸，np.nan代表不指定，可以使用 `Bounds.bounded` 来部分指定

        Returns
        -------
        ret
            扩张边界后的网格

        Notes
        -----
        特别地，可以在 `cells` 中通过指定负数长度的 :class:`SizedDummy` 来代表在对应方向上删除若干网格，例如

        >>> # 删除 -x 方向边界上的 3 个网格
        >>> model_mesh.expand_cells(cells=Bounds.bounded(xmin=SizedDummy(-3)))
        """
        assert isinstance(self.mesh, TensorMesh), (
            f'`{ModelledMesh.expand_cells.__name__}` supports only `TensorMesh` for now.'
        )

        ndim = self.ndim

        if np.ndim(n_cells) == 0:
            n_cells = (np.asarray([-1, 1] * ndim) * n_cells).view(Bounds)
        else:
            n_cells = Bounds([0] * 2 * ndim).override(by=n_cells, inplace=True)

        if np.ndim(ratio) == 0:
            ratio = np.full(2 * ndim, ratio).view(Bounds)
        else:
            ratio = Bounds([1] * 2 * ndim).override(by=ratio, inplace=True)

        old_h = self.mesh.h

        if cells is None:
            cells = []
            for axis in range(ndim):
                a_exps = ratio.get(axis)
                a_cells = n_cells.get(axis).astype(np.intp)

                old_h_axis = old_h[axis]

                if a_cells[0] < 0:
                    # 扩张网格
                    base = old_h_axis[0]
                    cells.append(base * np.logspace(1, a_cells[0], num=a_cells[0], base=a_exps[0])[::-1])
                elif a_cells[0] == 0:
                    cells.append(np.array([], dtype=bool))
                else:
                    # 收缩网格
                    cells.append(DeleteAndExtend(a_cells[0]))

                if a_cells[1] > 0:
                    # 扩张网格
                    base = old_h_axis[-1]
                    cells.append(base * np.logspace(1, a_cells[1], num=a_cells[1], base=a_exps[1]))
                elif a_cells[1] == 0:
                    cells.append(np.array([], dtype=bool))
                else:
                    # 收缩网格
                    cells.append(DeleteAndExtend(-a_cells[1]))

        new_h_spec = []
        delta_origin = np.zeros_like(self.origin)
        for axis in range(ndim):
            ncs, pcs = cells[axis * 2], cells[axis * 2 + 1]
            old_h_axis = old_h[axis]
            current_h = [ncs, old_h_axis, pcs]

            delta_origin[axis] -= np.sum(ncs)
            if isinstance(ncs, DeleteAndExtend):
                delta_origin[axis] += np.sum(ncs.clip(current_h[1], pos=0))
                current_h[1] = ncs.delete(current_h[1], pos=0)

            if isinstance(pcs, DeleteAndExtend):
                current_h[1] = pcs.delete(current_h[1], pos=-1)

            new_h_spec.append(current_h)

        return self.reset_tensor_mesh_specs(
            new_h_spec,
            self.origin + delta_origin,
            _skip_interior_checks=True
        )

    def reset_tensor_mesh_specs(self, hs=None, origin=None, *, _skip_interior_checks=False):
        """重新设置结构化网格的网格参数（每个轴方向上的网格间距）

        同时将旧网格上的模型参数重映射到新网格上

        Parameters
        ----------
        hs
            新的网格参数定义
        origin
            新的网格原点
        _skip_interior_checks
            指示是否跳过内部网格校验，检查新的网格参数是否导致了已有的非边界网格尺寸发生变化

        Returns
        -------
        ret
            更新定义后的网格

        Notes
        -----
        由于需要重新映射参数，因此本函数只支持在边界增减网格或修改边界网格，
        如果对内部的网格进行修改则会导致模型重映射给出错误的结果
        """
        old_h = self.mesh.h
        new_h = [np.concatenate(h) for h in hs]

        new_mesh = TensorMesh(new_h, origin=origin)

        # 计算新旧网格的三轴网格数
        old_mesh_shape = [len(h) for h in old_h]
        new_mesh_shape = [len(h) for h in new_h]

        # 构造新旧网格的张量有效网格掩码
        old_active_mask = self.active_mask
        old_mesh_active_ = old_active_mask.reshape(old_mesh_shape, order='F')
        old_mesh_active = np.zeros(old_mesh_shape, dtype=bool, order='F')
        new_mesh_active = np.zeros(new_mesh_shape, dtype=bool, order='F')

        # 计算新旧网格的三轴网格数变化量
        axis_bounds = np.asarray([(h[0].shape[0], h[-1].shape[0]) for h in hs], dtype=object)

        # 计算旧网格在新网格中的三轴相交位置
        new_ind_mask_bound = axis_bounds.copy()
        new_ind_mask_bound[:, 1] *= -1
        new_ind_mask_bound[axis_bounds <= 0] = None
        if np.all(new_ind_mask_bound == None):
            new_ind_mask_slices = slice(None)
        else:
            new_ind_mask_slices = tuple(slice(*b) for b in new_ind_mask_bound)

        # 计算新网格在旧网格中的三轴相交位置
        old_ind_mask_bound = axis_bounds.copy()
        old_ind_mask_bound[:, 0] *= -1
        old_ind_mask_bound[axis_bounds >= 0] = None
        if np.all(old_ind_mask_bound == None):
            old_ind_mask_slices = slice(None)
        else:
            old_ind_mask_slices = tuple(slice(*b) for b in old_ind_mask_bound)

        if not _skip_interior_checks:
            # 检查新的网格规格是否修改了内部网格的尺寸
            for i, (old, old_mask, new, new_mask) in enumerate(zip(
                    old_h, old_ind_mask_bound, new_h, new_ind_mask_bound
            )):
                if np.any(old[slice(*old_mask)][1:-1] != new[slice(*new_mask)][1:-1]):
                    raise ValueError(
                        f'Modification of interior cells detected on axis {i},'
                        f' which will lead to undefined behavior when remapping model.'
                        f' To force continuing the process,'
                        f' try disable this check by setting `_skip_checks=True`.'
                    )

        new_mesh_active[new_ind_mask_slices] = True
        old_mesh_active[old_ind_mask_slices] = old_mesh_active_[old_ind_mask_slices]

        old_ind_mask = old_mesh_active.ravel(order='F')[old_active_mask]
        if np.all(old_ind_mask):
            old_ind_mask = slice(None)

        # 先使旧空间区域为新空间的有效网格，方便将旧网格上的complete_model映射为新网格的complete_model
        ret = ModelledMesh(new_mesh, ind_active=new_mesh_active.ravel(order='F'))

        models = {}
        for key in self:
            model = self.get_raw_model(key)
            if self.is_active_model(model):
                models[key] = model[old_ind_mask]  # active_model可能需要移除部分网格
            else:
                masked_old_complete_model = model.reshape(old_mesh_shape, order='F')[old_ind_mask_slices]
                models[key] = ret.map_to_complete(masked_old_complete_model.ravel(order='F'))

        # 重新绑定为实际的有效网格掩码
        new_mesh_active[new_ind_mask_slices] = old_mesh_active_[old_ind_mask_slices]
        ret.active_mask = new_mesh_active.ravel(order='F')

        for key, model in models.items():
            ret[key] = model

        if self.has_default_model:
            ret.set_active_model(self.default_key)

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


def compute_trimmed_length(width, h, precise):
    trimmed_cells = len(h)
    length = 0
    for i in range(len(h)):
        ci = h[i]
        length += ci
        if length > width:
            length -= ci
            trimmed_cells = i
            break

    h = h[trimmed_cells:]
    if precise:
        h[0] -= width - length
        length = width

    return h, width - length


def compute_exponential_cell_specs(width, base_cell_size, ratio, n_cells, precise):
    if not np.isnan(n_cells):
        if base_cell_size * n_cells > width:
            # 不需要增长网格大小也可以达到width宽度，转为使用等距网格
            warnings.warn(f'Too much cells specified ({int(n_cells)} * {base_cell_size} > {width}),'
                          f' using uniform cells.')
            ratio = 1
        if ratio == 1:
            # 要求等距网格
            base_cell_size = width / n_cells
        else:
            # 注意此处第一项并非 base_cell_size，因此需要补充首项，并增加项数
            ratio = solve_geometric_ratio(width + base_cell_size, n_cells + 1, base_cell_size)
            theoretical_width = ratio * (ratio ** n_cells - 1) / (ratio - 1) * base_cell_size
            assert np.allclose(theoretical_width, width), 'Oops! Failed to find increment exponent.'
    else:
        if ratio == 1:
            # 要求等距网格
            n_cells = np.ceil(width / base_cell_size)
        else:
            assert np.isnan(ratio) or ratio > 1, ('Exponentially increasing cell sizes'
                                                  ' requires an exponent larger than 1.')
            # 注意此处第一项并非 base_cell_size，因此需要补充首项，并减去多余的项数
            cells_needed = solve_geometric_terms_count(width + base_cell_size, ratio, base_cell_size) - 1
            n_cells = np.ceil(cells_needed)

            if precise:
                # 精确模式下，检查新的网格边界是否和预期边界重合
                theoretical_width = ratio * (ratio ** n_cells - 1) / (ratio - 1) * base_cell_size
                if not np.allclose(theoretical_width, width):
                    warnings.warn(
                        f'Setting `ratio={ratio}` is resulting mismatched mesh bounds under `precise` mode.'
                        f' Consider using `precise=False` or setting `n_cells`'
                        f' to acquire precisely aligned bounds.'
                    )

    h = np.logspace(1, n_cells, num=int(n_cells), base=ratio) * base_cell_size
    actual_width = np.sum(h)

    return h, actual_width - width


def solve_geometric_ratio(s_n, n, a):
    """高斯牛顿法求解等比数列 {:math:`a, aq, aq^2, aq^3...`} 的公比

    即求解 `q` ，使得

    :math:`a(1 - q ^ n) / (1 - q) = S_n`

    Parameters
    ----------
    s_n
        前n项和
    n
        总项数
    a
        首项

    Returns
    -------
    q
        公差
    """
    max_iters = 100
    r = s_n / a
    q = r ** (1 / n)
    for i in range(max_iters):
        p = q
        q_n_1 = q ** (n - 1)
        q = q - (q_n_1 * q - r * q + r - 1) / (n * q_n_1 - r)

        dq = abs((p - q) / q)
        if dq < 1e-6:
            break

    return q


def solve_geometric_terms_count(s_n, q, a):
    """求解等比数列 {:math:`a, aq, aq^2, aq^3...`} 的项数

    即求解 `n` ，使得

    :math:`a(1 - q ^ n) / (1 - q) = S_n`

    Parameters
    ----------
    s_n
        前n项和
    q
        公比
    a
        首项

    Returns
    -------
    n
        项数（可能不为整数）
    """
    return np.log(s_n / a * (q - 1) + 1) / np.log(q)


class DeleteAndExtend:
    def __init__(self, deletion, arr: Sequence = tuple()):
        """代表在特定方向上移除 `deletion` 个元素后追加 arr

        Parameters
        ----------
        deletion
            需要移除的元素数
        arr
            需要追加的元素
        """
        self.deletion = deletion
        self.arr = arr

    @property
    def shape(self):
        return (-self.deletion + len(self.arr),)

    def __iter__(self):
        yield from self.arr

    def __repr__(self):
        return f'{DeleteAndExtend.__name__}(deletion={self.deletion}, arr={self.arr})'

    def __array__(self):
        return np.asarray(self.arr)

    def clip(self, sequence, pos=0):
        if self.deletion == 0:
            return sequence[:0]

        if pos >= 0:
            return sequence[pos:pos + self.deletion]
        else:
            if pos == -1:
                slicer = slice(pos - self.deletion + 1, None)
            else:
                slicer = slice(pos - self.deletion + 1, pos + 1)

            return sequence[slicer]

    def delete(self, sequence, pos=0):
        if isinstance(sequence, (list, tuple)):
            def concat(seqs):
                return sum(seqs, type(sequence)())
        else:
            concat = np.r_.__getitem__

        if self.deletion == 0:
            return sequence

        if pos >= 0:
            if pos == 0:
                return sequence[self.deletion:]
            else:
                return concat((sequence[:pos], sequence[pos + self.deletion:]))
        else:
            if pos == -1:
                return sequence[:-self.deletion]
            else:
                return concat((sequence[:pos - self.deletion + 1], sequence[pos + 1:]))
