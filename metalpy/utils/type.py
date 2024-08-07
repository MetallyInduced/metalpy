from __future__ import annotations

import copy
import functools
import sys
import types
import warnings
from typing import Iterable, Callable, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

Self = Self

T = TypeVar('T')


class Dummy:
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Dummy) and self.name == other.name


undefined = Dummy('__undefined')


def stringify(e):
    return [str(i) for i in e]


def ensure_as_iterable(arg, excludes=None):
    if excludes is None:
        excludes = []
    elif not isinstance(excludes, Iterable):
        excludes = [excludes]

    if arg is None:
        return []
    for exclude in excludes:
        if isinstance(arg, exclude):
            return [arg]
    if not isinstance(arg, Iterable):
        return [arg]

    return arg


def get_or_default(dictionary, key, _default=None, default=None, supplier: Callable | None = None, remove=False):
    """获取键对应的值，若不存在则返回指定的默认值

    Parameters
    ----------
    dictionary
        映射
    key
        键
    _default
        默认值

        .. deprecated::
            使用default
    default
        默认值
    supplier
        默认值提供器，需要返回默认值且未指定default时，将尝试调用supplier获取返回值
    remove
        指示找到键值对时是否删除该元素

    Returns
    -------
    ret
        dictionary[key]，若`key`不存在且`default`非空则返回`default`，否则调用`supplier()`作为返回值
    """
    if key in dictionary:
        ret = dictionary[key]
        if remove:
            del dictionary[key]
    else:
        if default is None:
            default = _default
        if default is None and supplier is not None:
            default = supplier()
        ret = default

    return ret


def pop_or_default(dictionary, key, _default=None, default=None, supplier: Callable | None = None):
    """获取并移除键对应的值，若不存在则返回指定的默认值

    Parameters
    ----------
    dictionary
        映射
    key
        键
    _default
        默认值

        .. deprecated::
            使用default
    default
        默认值
    supplier
        默认值提供器，需要返回默认值且未指定default时，将尝试调用supplier获取返回值

    Returns
    -------
    ret
        dictionary[key]，若`key`不存在且`default`非空则返回`default`，否则调用`supplier()`作为返回值

    See Also
    --------
    get_or_default
    """
    return get_or_default(
        dictionary=dictionary,
        key=key,
        _default=_default,
        default=default,
        supplier=supplier,
        remove=True
    )


def not_none_or_default(val, _default=None, default=None, supplier: Callable | None = None):
    """判断指定值是否为空，非空则返回原值，否则返回指定的默认值

    Parameters
    ----------
    val
        待判断的值
    _default
        默认值

        .. deprecated::
            使用default
    default
        默认值
    supplier
        默认值提供器，需要返回默认值且未指定default时，将尝试调用supplier获取返回值

    Returns
    -------
    ret
        若`val`非空则返回`val`，若`val`为None且`default`非空则返回`default`，否则调用`supplier`作为返回值
    """
    if val is None:
        if default is None:
            default = _default
        if default is None and supplier is not None:
            default = supplier()
        return default
    else:
        return val


def get_first_key(dictionary):
    for k in dictionary:
        return k

    return None


def ensure_set_key(dictionary, key, value=undefined, *, transfer=None):
    while key in dictionary:
        if transfer is None:
            def transfer(x):
                return f'_{x}_'
        key = transfer(key)

    if undefined != value:
        dictionary[key] = value

    return key


def get_params_dict(**kwargs):
    return dict(kwargs)


def get_params(*args, **kwargs):
    return list(args), dict(kwargs)


def is_numeric_array(obj):
    dtype = getattr(obj, 'dtype', None)
    if dtype is None:
        return False

    # 用于进一步判断numpy和pandas的元素类型
    # torch数组只支持数值类型，因此不需要继续判断，用空字符串跳过后续分支
    dtype_name = getattr(dtype, 'name', '')

    non_numeric_keywords = [
        'str',  # numpy - 'str', pandas - 'string'
        'object'  # numpy, pandas - 'object'
    ]
    for k in non_numeric_keywords:
        if k in dtype_name:
            return False

    return True


def make_package_notification(pkg_name, reason, install=None, extra=None):
    sentences = [f'{reason}']

    if install is None:
        install = f'pip install {pkg_name}'

    if install is not False:
        sentences.extend([
            f'Consider install `{pkg_name}` with following command:',
            f'',
            f'  >>> {install}',
            f''
        ])

    if extra is not None:
        sentences.append(extra)

    return '\n'.join(sentences)


def notify_package(pkg_name, reason, install=None, extra=None):
    warnings.warn(make_package_notification(
        pkg_name,
        reason,
        install=install,
        extra=extra
    ))


def requires_package(pkg_name, reason, install=None, extra=None):
    raise ModuleNotFoundError(make_package_notification(
        pkg_name,
        reason,
        install=install,
        extra=extra
    ))


def copy_func(f: T, globals=None, module=None) -> T:
    """Based on:
        https://stackoverflow.com/a/6528148 (@Glenn Maynard)
        https://stackoverflow.com/a/13503277 (@unutbu)
        https://stackoverflow.com/a/49077211 (@Mad Physicist)
    """
    if globals is None:
        globals = copy.copy(f.__globals__)
    g = types.FunctionType(f.__code__, globals, name=f.__name__,
                           argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    if module is not None:
        g.__module__ = module
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g


def get_all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)])


class CachedProperty(object):
    """用于缓存计算开销较大的属性，仅在首次访问时计算并缓存结果

    不添加线程锁，线程不安全，但是可以被序列化

    TODO: 考察是否有办法检测被序列化，并在序列化时不序列化缓存的属性？

    References
    ----------
    Based on:
        https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76 (@pydanny)
    """
    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

        functools.wraps(func)(self)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

    def set(self, obj, val):
        obj.__dict__[self.func.__name__] = val

    def invalidate(self, obj):
        obj.__dict__.pop(self.func.__name__, None)
