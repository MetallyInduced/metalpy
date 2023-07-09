from __future__ import annotations

import os.path
from functools import cached_property
from typing import Union, Iterable

import numpy as np
import pyvista as pv

from metalpy.utils.dhash import dhash
from metalpy.utils.bounds import Bounds
from metalpy.utils.file import openable
from metalpy.utils.model import hash_model, split_models_in_memory, load_model_file, load_grouped_file, \
    extract_model_list_bounds, dhash_model
from . import Shape3D
from ..utils.mesh import is_inside_cuboid


def check_density(length, dx, nx, _default):
    if dx is not None:
        return dx
    elif nx is not None:
        return length / nx
    else:
        return _default


def check_scale(length, target_size, target_scale, _default=1):
    if target_scale is not None:
        return target_scale
    elif target_size is not None:
        return target_size / length
    else:
        return _default


def check_is_none(val, _default):
    if val is not None:
        return val
    else:
        return _default


def check_scalar_or_list(val, size, _default):
    if val is None:
        return [_default] * size
    elif not hasattr(val, '__iter__'):
        return [val] * size
    else:
        return list(val)


class Obj2(Shape3D):
    SUBDIVIDE_IN_MEMORY = 'memory'
    SUBDIVIDE_FILE = 'file'
    SUBDIVIDE_NONE = False

    def __init__(self, model: Union[str, pv.DataSet, None],
                 xsize=None, ysize=None, zsize=None, size=None,
                 xscale=None, yscale=None, zscale=None, scale=None,
                 surface_thickness=None, surface_range: float | list[float] | None = None,
                 subdivide: str | bool | None = SUBDIVIDE_NONE,
                 ignore_surface_check=False,
                 keep_original_model=True,
                 verbose=True):
        """定义一个基于通用格式的模型

        Parameters
        ----------
        model
            模型路径或PyVista模型对象 (当为None时不主动构造，用于内部的拷贝代码，非必要请勿使用)
        xsize, ysize, zsize, size
            模型尺寸，构造时自动将模型缩放到该尺寸下
        xscale, yscale, zscale, scale
            模型缩放比例，构造时自动根据该比例缩放模型
        surface_thickness
            表面厚度（向内），即surface_range=[-surface_range, 0]
        surface_range : array(2) or number
            表面距离范围，用于选定到表面有向距离在该范围内的网格，
            小于0表示模型内，大于0表示模型外，None表示不作判断，
            如果为单个数r，则自动设置为[0, r]或[r, 0]。
        subdivide
            是否模型细分（非通用）。
            SUBDIVIDE_IN_MEMORY 载入后根据点连接性来模型细分。
            SUBDIVIDE_FILE 对模型文件进行组分割（仅限obj文件）再载入。
        ignore_surface_check
            代表使用内置体素化算法时不检测模型封闭性
        keep_original_model
            代表是否保留最初导入的未分割的模型，如果为`False`则不会保留，to_polydata()等涉及获取原始模型的操作会通过pv.merge获得。
            适用于不需要再次访问原始模型的情形。
        verbose
            控制是否输出相关信息
        """
        super().__init__()

        self.ignore_surface_check = ignore_surface_check
        if surface_thickness is not None or surface_range is not None:
            if surface_range is not None:
                if isinstance(surface_range, Iterable):
                    self.surface_range = tuple(surface_range)
                else:
                    if surface_range >= 0:
                        self.surface_range = (0, surface_range)  # 0 ~ r
                    else:
                        self.surface_range = (surface_range, 0)  # r ~ 0
            else:
                if surface_thickness is not None:
                    self.surface_range = [-surface_thickness, 0]
                else:
                    self.surface_range = None
        else:
            self.surface_range = None

        if subdivide is True:
            subdivide = Obj2.SUBDIVIDE_IN_MEMORY

        models = None
        did_subdivision = False
        if model is not None:
            should_scale_be_inplace = False
            if openable(model):
                model_path = model
                should_scale_be_inplace = True  # 从文件导入的model的scale操作可以inplace

                if not subdivide:
                    model = load_model_file(model_path, verbose=verbose)
                    models = [model]
                elif subdivide == Obj2.SUBDIVIDE_IN_MEMORY:  # 导入后通过连通性分析分割模型
                    model = load_model_file(model_path, verbose=verbose)
                    models = split_models_in_memory(model, verbose=verbose)
                    did_subdivision = True
                elif subdivide == Obj2.SUBDIVIDE_FILE:  # 从文件层面切割得到的子模型文件导入到models
                    models = load_grouped_file(model_path, verbose=verbose)
                    if models is None:
                        raise NotImplementedError(
                            f"Model file splitter is not available for {os.path.splitext(model_path)[1]} file. \n"
                            f"Which is {model_path} .")
                    did_subdivision = True
                    model = None
                    keep_original_model = False  # 此时不存在原模型，无法保留
            else:
                # then model should be PyVista object
                if not subdivide:
                    models = [model]
                elif subdivide == Obj2.SUBDIVIDE_IN_MEMORY:  # 导入后通过连通性分析分割模型
                    models = split_models_in_memory(model, verbose=verbose)
                    did_subdivision = True
                    should_scale_be_inplace = True

            bounds = extract_model_list_bounds(models)
            xl, yl, zl = bounds[1::2] - bounds[0::2]

            size = check_scalar_or_list(size, 3, None)

            size[0] = check_is_none(xsize, size[0])
            size[1] = check_is_none(ysize, size[1])
            size[2] = check_is_none(zsize, size[2])
            xsize, ysize, zsize = size

            scale = check_scalar_or_list(scale, 3, 1)

            scale[0] = check_scale(xl, xsize, xscale, scale[0])
            scale[1] = check_scale(yl, ysize, yscale, scale[1])
            scale[2] = check_scale(zl, zsize, zscale, scale[2])

            if np.any(np.asarray(scale) != 1):
                for i in range(len(models)):
                    models[i] = models[i].scale(scale, inplace=should_scale_be_inplace)
                if did_subdivision and keep_original_model:
                    model = model.scale(scale, inplace=should_scale_be_inplace)

                bounds = extract_model_list_bounds(models)
                should_scale_be_inplace = True

            for m in models:
                m.translate(-bounds[::2], inplace=should_scale_be_inplace)
            if did_subdivision and keep_original_model:
                model = model.translate(-bounds[::2], inplace=should_scale_be_inplace)

        self.models = models

        if keep_original_model:
            if did_subdivision:
                self.model = model
            else:
                if models is not None:
                    self.model = models[0]

    @property
    def surface_only(self):
        return self.surface_range is not None

    @cached_property
    def model(self):
        """获取原始模型，
        如果构造函数指定`keep_original_model`则来自构造函数，
        否则基于细分的models进行merge获取
        """
        return pv.merge(self.models, merge_points=False)

    @property
    def has_original_model(self):
        """判断model属性是否存在缓存值（即不需要进行生成）
        """
        return Obj2.model.attrname in self.__dict__

    def clear_original_model_cache(self):
        """清空model属性的缓存值
        """
        del self.__dict__[Obj2.model.attrname]

    def get_model(self, copy):
        return self.model.copy(deep=copy)

    def place_impl(self, model, mesh_cell_centers, progress):
        p0, p1 = Bounds(model.bounds).as_corners()
        indices = is_inside_cuboid(mesh_cell_centers, p0, p1 - p0)

        mesh_cell_centers = mesh_cell_centers[indices]
        active_indices = indices[indices]

        poly = pv.PolyData(mesh_cell_centers)
        mesh = pv.UnstructuredGrid(poly.cast_to_unstructured_grid())
        surface = model.extract_surface()

        if self.surface_range:
            surface_distance = mesh.compute_implicit_distance(surface)
            smin, smax = self.surface_range

            dist = surface_distance['implicit_distance']

            if smax is not None:
                active_indices &= dist <= smax

            if smin is not None:
                active_indices &= dist >= smin
        else:
            selection = mesh.select_enclosed_points(surface, tolerance=0.0, check_surface=not self.ignore_surface_check)
            active_indices = selection['SelectedPoints'].view(np.bool_)

        indices[indices] = active_indices

        if progress is not None:
            progress.update(1)

        return indices

    def do_place(self, mesh_cell_centers, progress):
        indices = None
        for model in self.models:
            ret = self.place_impl(model, mesh_cell_centers, progress)
            if indices is None:
                indices = ret
            else:
                indices = np.logical_or(indices, ret)

        return indices

    def do_hash(self):
        n_samples = np.max((2, 10 // len(self.models)))
        surface_range = self.surface_range if self.surface_range is not None else (None,)
        return hash((*(hash_model(m, n_samples) for m in self.models),
                     *surface_range,
                     ))

    def __dhash__(self):
        n_samples = np.max((2, 10 // len(self.models)))
        return dhash(super().__dhash__(),
                     self.surface_range,
                     *(dhash_model(m, n_samples) for m in self.models),
                     )

    def do_clone(self, deep=True):
        ret = Obj2(
            None,
            surface_range=self.surface_range
        )
        if not deep and self.has_original_model:
            ret.model = self.get_model(copy=False)

        ret.models = [m.copy(deep=deep) for m in self.models]

        return ret

    @property
    def local_bounds(self):
        ret = Bounds(extract_model_list_bounds(self.models))
        if self.surface_range:
            smax = self.surface_range[1]
            if smax is not None:
                ret.expand(increment=smax)

        return ret

    def to_local_polydata(self):
        return self.get_model(copy=True)

    @property
    def volume(self):
        return sum([m.volume for m in self.models])

    @property
    def area(self):
        return sum([m.area for m in self.models])

    @property
    def n_tasks(self):
        return len(self.models)

    @property
    def progress_manually(self):
        return True
