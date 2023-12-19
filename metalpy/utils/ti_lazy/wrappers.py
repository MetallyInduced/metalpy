from __future__ import annotations

import functools
import inspect
import math
from typing import TypeVar, Callable, Generic

import numpy

from metalpy.utils.type import copy_func
from . import ti as ti_dummy

FuncT = TypeVar('FuncT', bound=Callable)


def lazy_kernel(fn):
    cache_key = '_kernel_cache'

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        kernel = getattr(fn, cache_key, None)
        if kernel is None:
            kernel = fn()
            setattr(fn, cache_key, kernel)

        return kernel(*args, **kwargs)

    return wrapper


class LazyTaichiInstance(Generic[FuncT]):
    def __init__(self, fn: FuncT, func=False, kernel=False, is_real_function=None):
        self._original: FuncT = fn
        self.func: FuncT = lazy_kernel(lambda: ti_make_func(fn, is_real_function=is_real_function)) if func else None
        self.kernel: FuncT = lazy_kernel(lambda: ti_make_kernel(fn)) if kernel else None

    def __call__(self, *args, **kwargs):
        return self._original(*args, **kwargs)


def ti_lazy_kernel(fn: FuncT) -> FuncT | LazyTaichiInstance[FuncT]:
    return LazyTaichiInstance(fn, kernel=True)


def ti_lazy_func(fn: FuncT, is_real_function=None) -> FuncT | LazyTaichiInstance[FuncT]:
    return LazyTaichiInstance(fn, func=True, is_real_function=is_real_function)


def ti_make_func(fn, is_real_function=None):
    """将python函数转化为 `ti.func`

    可配合 `metalpy.utils.taichi_lazy.lazy_kernel` 使用，实现将element-wise的函数转换为 `ti.func`
    """
    from metalpy.utils.taichi import ti_func

    return ti_func(ti_annotate(fn), is_real_function=is_real_function)


def ti_make_kernel(fn):
    """将python函数转化为 `ti.kernel`

    同 `ti.kernel` 要求，所有参数需要给出注解 (未注解默认未 `ti.template` )
    """
    from metalpy.utils.taichi import ti_kernel

    return ti_kernel(ti_annotate(fn))


def ti_annotate(fn):
    """将python函数的参数标注转换为对应类型，默认 `ti.template` (指示函数参数按引用传递)

    将python函数中对 `math` 和 `numpy` 的引用替换为 `ti`
    """
    import taichi as ti

    fn = copy_func(fn)

    spec = inspect.getfullargspec(fn)
    for arg in spec.args:
        annotation = fn.__annotations__.get(arg, None)
        if annotation is None:
            fn.__annotations__[arg] = ti.template()
        else:
            if isinstance(annotation, str):
                annotation = eval(annotation, fn.__globals__)  # TODO: 可能导致安全隐患? 限制允许的类型?
            fn.__annotations__[arg] = ti_dummy.lang.to_taichi_type(annotation)

    for key in fn.__globals__.keys():
        var = fn.__globals__[key]
        if var is math or var is numpy or var is ti_dummy:
            fn.__globals__[key] = ti  # 可能会导致一些问题?

    _ti_import_math()

    return fn


def _ti_import_math():
    """将部分 `ti.math` 的内容引入到 `ti` 空间中，
    与 `math` 和 `numpy` 保持一致，
    以支持在 `kernel` 或 `func` 中使用 `math` 和 `numpy` 的函数，
    并在 `ti_annotate` 中自动完成转换

    注：部分函数无法完全重合，只支持基本的函数使用
    """
    flag_key = '_kernel_cache'

    if not getattr(_ti_import_math, flag_key, False):
        import taichi as ti

        for key in dir(ti.math):
            if key == 'inf':
                x = 1
            if key.startswith('__'):
                continue
            if getattr(math, key, None) is None:
                continue
            if getattr(numpy, key, None) is None:
                if key.startswith('a'):  # atan2, acos, asin -> arctan2, arccos, arcsin
                    arc_key = 'arc' + key[1:]
                    if getattr(numpy, arc_key, None) is None:
                        continue
                    setattr(ti, arc_key, getattr(ti.math, key))
                else:
                    continue
            if getattr(ti, key, None) is not None:
                continue

            setattr(ti, key, getattr(ti.math, key))

        setattr(_ti_import_math, flag_key, True)
