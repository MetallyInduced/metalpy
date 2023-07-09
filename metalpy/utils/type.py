from __future__ import annotations

from typing import Iterable, Callable


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
