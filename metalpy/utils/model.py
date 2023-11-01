import os.path
import pickle
from enum import Enum
from typing import Iterable, Literal, Union, Any

import numpy as np
import pyvista as pv
import tqdm

from .algos import QuickUnion, ConnectedTriangleSurfaces
from .dhash import _hash_array
from .file import make_cache_file, make_cache_directory
from .hash import hash_numpy_array
from .obj_splitter import ObjSplitter
from .string import parse_axes_labels
from .time import Timer


DataSetLike = Union[pv.DataSet, pv.MultiBlock]
DataSetLikeType = (pv.DataSet, pv.MultiBlock)


def hash_model(model, n_samples=10):
    return hash_numpy_array(model.points, n_samples)


def dhash_model(model, n_samples=10):
    return _hash_array(model.points, n_samples)


def as_pyvista_array(arr):
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.int8)

    return arr


def extract_model_list_bounds(model_list: list[pv.DataSet]):
    bounds = None
    for m in model_list:
        if bounds is None:
            bounds = np.asarray(m.bounds)
            continue
        bounds[1::2] = np.max([bounds[1::2], m.bounds[1::2]], axis=0)
        bounds[0::2] = np.min([bounds[0::2], m.bounds[0::2]], axis=0)

    return bounds


def merge_as_multiblock(model_list):
    mb = pv.MultiBlock()
    for m in model_list:
        mb.append(m)

    return mb


def load_model_file(model_file, verbose=True):
    model_name = os.path.basename(model_file)
    cache_path = make_cache_file(model_name + '.vtk')

    if os.path.exists(cache_path):
        if verbose:
            print(f'Loading cached model from {cache_path}...')
        model_file = cache_path

    reader = pv.get_reader(model_file)

    if verbose:
        reader.show_progress()

    timer = Timer()
    with timer:
        model = reader.read()

    if timer.elapsed > 5 and cache_path != model_file:
        if verbose:
            print(f'Saving loaded model to {cache_path}...')
        model.save(cache_path, binary=True)

    return model


def load_grouped_file(model_file, verbose=True):
    if os.path.splitext(model_file)[1] != '.obj':
        return None  # 目前只支持obj文件

    splitter = ObjSplitter(model_file)
    model_name = os.path.basename(model_file)
    cache_path = make_cache_directory(os.path.splitext(model_name))
    if not os.path.exists(cache_path):
        splitter.split_by_group(cache_path)

    models = []
    for sub_model_file in os.listdir(cache_path):
        sub_model_path = os.path.join(cache_path, sub_model_file)
        models.append(load_model_file(sub_model_path))

    return models


def split_models_in_memory(model, verbose=True):
    cache = make_cache_file(f'model_division/{dhash_model(model)}.sub')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            if verbose:
                print(f'Loading split models from {cache}...')
            ret = pickle.load(f)
    else:
        models = split_models_in_memory_pointwisely(model, verbose=verbose)
        ret = []

        if verbose:
            progress = tqdm.tqdm(total=model.n_faces, desc='Splitting models by edge connectivity')
        else:
            progress = None

        for m in models:
            ms = split_models_in_memory_edgewisely(m, verbose=verbose, progress=progress)
            ret.extend(ms)

        with open(cache, 'wb') as f:
            if verbose:
                print(f'Saving split models to {cache}...')
            pickle.dump(models, f)

    return ret


def split_models_in_memory_edgewisely(model, verbose=True, progress=None):
    points = ConnectedTriangleSurfaces()
    model_faces = model.faces
    i = 0
    faces_size = len(model_faces)

    if verbose and progress is None:
        progress = tqdm.tqdm(total=faces_size, desc='Splitting models by edge connectivity')

    while i < faces_size:
        nv = model_faces[i]
        pts = model_faces[i + 1:i + nv + 1]
        i = i + nv + 1
        points.add(pts)
        if progress is not None:
            progress.update(1)

    models = []
    for g in points.get_groups():
        indices = np.asarray(list(g))
        sub_model = model.extract_points(indices, adjacent_cells=False)
        sub_model.clear_data()
        models.append(sub_model.extract_surface())

    return models


def split_models_in_memory_pointwisely(model, verbose=True):
    """
    按点连通性来划分子几何体，对一些情况无法适用
    """
    unions = QuickUnion(model.points.shape[0])
    model_faces = model.faces
    i = 0
    faces_size = len(model_faces)

    progress = None
    if verbose:
        progress = tqdm.tqdm(total=faces_size, desc='Splitting models by point connectivity')

    while i < faces_size:
        nv = model_faces[i]
        pts = model_faces[i + 1:i + nv + 1]
        i = i + nv + 1
        p1 = pts[0]
        for p2 in pts[1:]:
            unions.connect(p1, p2)

        if progress is not None:
            progress.update(nv + 1)

    if verbose:
        progress.close()

    unions.collapse(verbose)
    groups = np.unique(unions.unions)

    models = []

    if verbose:
        groups = tqdm.tqdm(groups, desc='Extracting sub-models')

    for g in groups:
        indices = np.argwhere(unions.unions == g).squeeze()
        sub_model = model.extract_points(indices, adjacent_cells=False)
        sub_model.clear_data()
        models.append(sub_model.extract_surface())

    return models


class ModelTranslation:
    def __init__(self, offset, inplace=False):
        self.offset = offset
        self.inplace = inplace

    def __call__(self, mesh):
        return pv_ufunc_apply(mesh, self._translate, inplace=self.inplace)

    def _translate(self, x):
        return x.translate(self.offset, inplace=True)


def pv_ufunc_apply(obj: DataSetLike, fn, *, inplace=True):
    """对PyVista对象执行指定函数。

    如果目标为 :class:`pyvista.DataSet` 及其子类，则直接将fn作用于obj；
    如果目标为 :class:`pyvista.MultiBlock` 则遍历其所有成员递归执行pv_ufunc_apply。

    Parameters
    ----------
    obj
        需要应用函数 `fn` 的PyVista对象
    fn
        需要应用到 `obj` 对象上的处理函数。
        语义上要求为 `inplace` 操作，具体 `inplace` 行为由 `inplace` 参数控制
    inplace
        指示操作是否直接在目标对象上进行。
        若为 `False` ，则会在所有对象的拷贝上应用 `fn` ，然后重新拼装完整对象。

    Returns
    -------
    obj
        obj自身，如果 `inplace` 为False，则为拷贝的新对象
    """
    if isinstance(obj, pv.MultiBlock):
        if inplace:
            for k in obj.keys():
                pv_ufunc_apply(obj[k], fn, inplace=inplace)
            return obj
        else:
            return pv.MultiBlock(
                {k: pv_ufunc_apply(obj[k], fn, inplace=inplace) for k in obj.keys()}
            )
    else:
        if not inplace:
            obj = obj.copy()
        fn(obj)
        return obj


def pv_ufunc_visit(obj: DataSetLike, fn):
    """遍历PyVista对象并执行指定函数，不会进行拷贝，如果 `fn` 产生了修改，则会直接修改 `obj`。

     即 `pv_ufunc_apply` 的强制 `inplace` 版本，语义上要求 `fn` 不会对DataSet进行修改，只进行遍历获取信息。

    Parameters
    ----------
    obj
        需要应用函数 `fn` 的PyVista对象
    fn
        需要应用到 `obj` 对象上的处理函数
    """
    pv_ufunc_apply(obj, fn, inplace=True)


def pv_ufunc_map(obj: DataSetLike, fn):
    """遍历PyVista对象并执行映射

    Parameters
    ----------
    obj
        需要应用函数 `fn` 的PyVista对象
    fn
        需要应用到 `obj` 对象上的映射
    """
    mapped = []

    def wrapper(dataset):
        mapped.append(fn(dataset))

    pv_ufunc_visit(obj, wrapper)

    return mapped


class DataAssociation(Enum):
    Cell = 'cell'
    Point = 'point'
    Row = 'row'

    Field = 'field'

    def __str__(self):
        return self.value

    @property
    def field_count(self):
        if self == DataAssociation.Field:
            return None
        else:
            return f'n_{self}s'

    @property
    def field_collection(self):
        return f'{self}_data'

    @property
    def field(self):
        return f'{self}s'

    def get(self, dataset):
        return getattr(dataset, self.field)

    def set(self, dataset, info):
        setattr(dataset, self.field, info)

    def get_count(self, dataset):
        return getattr(dataset, self.field_count)

    def get_collection(self, dataset):
        return getattr(dataset, self.field_collection)

    def get_data(self, dataset, key):
        return self.get_collection(dataset)[key]

    def set_data(self, dataset, key, val):
        self.get_collection(dataset).__setitem__(key, val)


def pv_ufunc_assign(obj: pv.DataSet,
                    data_type: DataAssociation,
                    key, val,
                    inplace=True,
                    set_active=True):
    """赋值给对象的数据字段，对MultiBlock则作用于其中的每个对象

    Parameters
    ----------
    obj
        待赋值对象
    data_type
        待赋值数据类型，例如 "`cell`", "`point`", "`field`"
    key
        待赋值字段类型
    val
        值，如果为单个值，则会通过np.full展开到对应字段长度
    inplace
        指示操作是否直接在目标对象上进行
    set_active
        指示是否将新赋予的值设置为默认向量

    Returns
    -------
    obj
        obj自身，如果`inplace`为False，则为拷贝的新对象
    """
    if inplace is False:
        obj = obj.copy()

    val = as_pyvista_array(val)

    def ufunc(x):
        element_count = data_type.get_count(x)

        if val.ndim == 0:
            _val = np.full(element_count, val)
        else:
            _val = val

        data_type.set_data(x, key, _val)

        if set_active:
            x.set_active_scalars(key)

        return x

    pv_ufunc_apply(obj, ufunc, inplace=inplace)


AxesLabels = Iterable[Union[Literal['x', 'y', 'z'], int]]


def reset_model_axes(model, new_axes: AxesLabels, inplace=False):
    new_axes = parse_axes_labels(new_axes, max_length=3)
    return reset_model_axes_unsafe(model=model, new_axes=new_axes, inplace=inplace)


def reset_model_axes_unsafe(model, new_axes: list[int], inplace=False):
    if not inplace:
        model = model.copy(deep=True)

    original_axes = []
    sub_axes = []
    for i, a in enumerate(new_axes):
        if i != a:
            original_axes.append(i)
            sub_axes.append(a)

    model.points[:, sub_axes] = model.points[:, original_axes]

    return model


def convert_model_dict_to_multiblock(model_dict: dict[Any, pv.DataSet], root=None):
    """将模型字典转换为pv.MultiBlock实例，支持嵌套字典

    Parameters
    ----------
    model_dict
        模型字典
    root
        基础MultiBlock实例

    Returns
    -------
    multi_block
        从模型字典构造的嵌套MultiBlock
    """
    if root is None:
        root = pv.MultiBlock()

    for k, v in model_dict.items():
        if isinstance(v, dict):
            v = convert_model_dict_to_multiblock(v)
        root[k] = v

    return root
