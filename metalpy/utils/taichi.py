from __future__ import annotations

import copy
import functools
import inspect
import os
import warnings
from typing import Sized

import numpy as np
import taichi as ti
from taichi.lang.kernel_impl import _kernel_impl
from taichi.lang.util import to_taichi_type, to_numpy_type

from .file import make_cache_directory, make_cache_file

ti_cache_prefix = 'taichi_cache'

ti_default_args = {
    'arch': ti.cpu,
    'offline_cache': True,
    'offline_cache_file_path': make_cache_directory(ti_cache_prefix),
}
ti_args = copy.deepcopy(ti_default_args)
ti_inited = False

ti_size_t = ti.i32
ti_size_dtype = to_numpy_type(ti_size_t)
ti_size_max = np.iinfo(ti_size_dtype).max


def ti_prepare(**kwargs) -> bool:
    """修改用于初始化taichi上下文的配置参数

    Parameters
    ----------
    kwargs
        需要修改的参数，以关键字参数形式指定

    Returns
    -------
    conflicted
        指示新的配置是否与已有配置冲突
    """
    conflicts = False
    for k, v in kwargs.items():
        if k not in ti_args or ti_args[k] != v:
            conflicts = True
        ti_args[k] = v
    return conflicts


def ti_init_once():
    global ti_inited
    if not ti_inited:
        ti_reinit()


def ti_reinit():
    global ti_inited
    ti.init(**ti_args)
    ti_inited = True


def ti_reset():
    """重置taichi的配置参数，使得下次调用涉及kernel的函数时会重新初始化

    同时也用于解决taichi目前没有gc导致内存泄露的问题
    taichi-dev/taichi#6803: Improve garbage collection
    https://github.com/taichi-dev/taichi/issues/6803

    Notes
    -----
        reset会在下次init时生效，因此不会立刻解决内存泄露的问题
    """
    global ti_inited, ti_args
    ti_args = copy.deepcopy(ti_default_args)
    ti_inited = False


class WrappedTaichiContext:
    def __init__(self, **kwargs):
        self.backup = ti_args.copy()
        self.kwargs = kwargs

    def __enter__(self):
        ti_prepare(**self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ti_prepare(**self.backup)


def ti_config(**kwargs):
    return WrappedTaichiContext(**kwargs)


def ti_arch(arch):
    arch_type = type(ti.cpu)
    if isinstance(arch, str):
        return getattr(ti, arch)
    elif isinstance(arch, arch_type):
        return arch
    elif isinstance(arch, list) and isinstance(arch[0], arch_type):
        return arch
    else:
        raise ValueError(f"Unknown architecture, "
                         f"should be one of 'cpu', 'gpu' or specific backends like 'cuda' or 'vulkan'. ")


def ti_func(fn, is_real_function=None):
    if is_real_function is None:
        return ti.func(fn)
    else:
        return ti.func(fn, is_real_function=is_real_function)


def ti_real_func(fn):
    return ti_func(fn, is_real_function=True)


def ti_kernel(fn):
    ti_wrapped = _kernel_impl(fn, level_of_class_stackframe=3)

    @functools.wraps(ti_wrapped)
    def auto_init_wrapped(*args, **kwargs):
        ti_init_once()  # 完成初始化
        return ti_wrapped(*args, **kwargs)

    return auto_init_wrapped


def ti_data_oriented(cls):
    return ti.data_oriented(cls)


def ti_ndarray(dtype, shape):
    ti_init_once()
    return ti.ndarray(dtype, shape)


def ti_zeros_like(arr, *, dtype=None):
    ti_init_once()
    if dtype is None:
        dtype = to_taichi_type(arr.dtype)
    return ti.ndarray(dtype, arr.shape)


def ti_root():
    ti_init_once()
    return ti.root


def ti_cfg():
    ti_init_once()
    return ti.cfg


def ti_field(dtype,
             shape=None,
             order=None,
             name="",
             offset=None,
             needs_grad=False,
             needs_dual=False):
    ti_init_once()
    dtype = to_taichi_type(dtype)
    return ti.field(dtype, shape, order, name, offset, needs_grad, needs_dual)


class WrappedFieldsBuilder:
    def __init__(self):
        self.fields_builder = ti.FieldsBuilder()
        self.snode_tree = None

    def finalize(self, raise_warning=True):
        self.snode_tree = self.fields_builder.finalize(raise_warning)
        return self.snode_tree

    def destroy(self):
        if self.snode_tree is not None:
            self.snode_tree.destroy()

    def place_dense(self, shape, *arr_types, axes=None):
        if axes is None:
            if isinstance(shape, Sized):
                ndim = len(shape)
            else:
                ndim = 1
            axes = ti.axes(*list(range(ndim)))
        arr_types = tuple(wrap_type_as_ti_field(t) for t in arr_types)
        self.dense(axes, shape).place(*arr_types)

        if len(arr_types) == 1:
            return arr_types[0]
        else:
            return arr_types

    def place_dense_like(self, arr, *, dtype=None, axes=None):
        if dtype is None:
            dtype = arr.dtype

        return self.place_dense(arr.shape, to_taichi_type(dtype), axes=axes)

    def __getattr__(self, name):
        return getattr(self.fields_builder, name)

    def __enter__(self) -> 'WrappedFieldsBuilder' | ti.FieldsBuilder:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()


def wrap_type_as_ti_field(t):
    if not isinstance(t, ti.Field):
        return ti_field(t)
    return t


def ti_FieldsBuilder():
    ti_init_once()
    return WrappedFieldsBuilder()


def ti_test_snode_support() -> bool:
    """测试cuda版本是否支持SNode和FieldsBuilder

    如果不支持，则需要使用动态数组ti.ndarray来代替

    Returns
    -------
    ret
        当前显卡cuda版本是否支持SNode和FieldsBuilder
    """
    ti_init_once()
    snode_support_indicator_cache = make_cache_file(f'{ti_cache_prefix}/snode_support')
    if os.path.exists(snode_support_indicator_cache):
        with open(snode_support_indicator_cache, 'r') as f:
            return f.read() == '1'
    try:
        with ti_FieldsBuilder() as builder:
            dummy = ti_field(ti.f64)
            builder.dense(ti.ij, (1, 1)).place(dummy)
            builder.finalize()
            _ = dummy.to_numpy()
            ret = True
    except RuntimeError as e:
        ret = False

    with open(snode_support_indicator_cache, 'w') as f:
        f.write('1' if ret else '0')

    return ret


def ti_ndarray_from(arr, sdim=0, dtype=None, field=False):
    """从其他数组类型创建Taichi ndarray

    Parameters
    ----------
    arr
        外部数组
    sdim
        结构维度，0代表Scalar，1代表Vector，2代表Matrix
    dtype
        指定taichi数组的数据类型，支持使用taichi或numpy的dtype标识
    field
        指示是否构建ti.field实例，若为False则构建ti.ndarray实例

    Returns
    -------
    ret
        返回结构维度为sdim的Taichi ndarray，且内容与arr相同
    """
    container = ti_ndarray_like(arr, sdim=sdim, dtype=dtype, field=field)
    copy_from(container, arr)
    return container


def ti_ndarray_like(arr, sdim=0, dtype=None, field=False, builder: ti.FieldsBuilder = None):
    """从其他数组类型创建相同尺寸的Taichi ndarray

    Parameters
    ----------
    arr
        外部数组
    sdim
        结构维度，0代表Scalar，1代表Vector，2代表Matrix
    dtype
        指定taichi数组的数据类型，支持使用taichi或numpy的dtype标识
    field
        指示是否构建ti.field实例，若为False则构建ti.ndarray实例
    builder
        指定用于分配数据的builder

    Returns
    -------
    ret
        返回结构维度为sdim的Taichi ndarray

    Notes
    -----
    field和ndarray区别参见：https://docs.taichi-lang.cn/docs/ndarray/
    """
    ti_init_once()

    if dtype is None:
        dtype = arr.dtype

    dt = to_taichi_type(dtype)
    if sdim == 0:
        shape = arr.shape
    elif sdim == 1:
        dt = ti.types.vector(arr.shape[-1], dt)
        shape = arr.shape[:-1]
    elif sdim == 2:
        dt = ti.types.matrix(*arr.shape[-sdim:], dt)
        shape = arr.shape[:-sdim]
    else:
        raise ValueError(f"Unsupported structure dim: {sdim}")

    if builder is None:
        if field:
            container = ti.field(dtype=dt, shape=shape)
        else:
            container = ti.ndarray(dtype=dt, shape=shape)
    else:
        container = ti.field(dt)
        builder.dense(ti.axes(*range(len(shape))), shape).place(container)

    return container


def copy_from(ti_arr, ext_arr):
    typeinfo = type(ext_arr)
    typename = typeinfo.__name__
    if typename == 'ndarray':
        ti_arr.from_numpy(ext_arr)
    else:
        raise ValueError(f"Unsupported array type: {typename}")
    # taichi暂未为ndarray实现这些来源
    # elif typeinfo.__module__ == 'torch':
    #     container.from_torch(arr)
    # elif typeinfo.__module__ == 'paddle':
    #     container.from_paddle(arr)


def ti_get_raw_function(kernel):
    raw = getattr(kernel, 'func', None) or getattr(kernel, '_primal', None) or kernel
    raw = getattr(raw, 'func', None) or raw

    return raw


def wrappable_for_ti_func(annotator):
    @functools.wraps(annotator)
    def wrapper(actual_func):
        # 用于处理ti.func或ti.kernel包装的函数
        # 这两类包装主要将实际功能函数添加到函数对象的__dict__中供调用
        funcs = []
        if getattr(actual_func, '_is_taichi_function', False):
            funcs.extend(['func'])
        elif getattr(actual_func, '_is_wrapped_kernel', False):
            funcs.extend(['_primal', '_adjoint'])

        ret = annotator(actual_func)

        for k in funcs:
            setattr(ret, k, annotator(getattr(ret, k)))

        return ret

    return wrapper


def check_contiguous(arr, name=None):
    typename = type(arr).__name__
    if typename == 'ndarray':
        import numpy as np
        if not arr.flags.contiguous:
            if name is None:
                loc = ''
            else:
                loc = f' at `{name}`'
            warnings.warn(f'Uncontiguous array detected{loc}.'
                          f' Copying to contiguous memory,'
                          f' which will consume extra memory.',
                          stacklevel=3)
            return np.ascontiguousarray(arr)
    else:
        # TODO: Torch、Paddle之类也可以检查？
        pass

    return arr


@wrappable_for_ti_func
def ensure_contiguous(func):
    """用于保证输入数组为连续数组（Taichi要求kernel的输入为连续数组）

    Parameters
    ----------
    func
        待包装的kernel函数

    Returns
    -------
    ret
        会自动查询输入的参数是否为连续数组，若否则拷贝一份到连续内存上

    Notes
    -----
    可能会导致引用数组地址变化，如果用于接受输出的数组被转换，则可能会导致输出到错误的位置
    """
    specs = inspect.getfullargspec(ti_get_raw_function(func))

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        n_args = len(args)
        n_varargs = len(specs.args) - n_args
        arg_names = specs.args + ([f'__vararg{i}' for i in range(-n_varargs)] if n_varargs < 0 else [])
        args = tuple(check_contiguous(arg, name=name) for arg, name in zip(args, arg_names))
        kwargs = {k: check_contiguous(v, name=k) for k, v in kwargs.items()}
        func(*args, **kwargs)

    return wrapper
