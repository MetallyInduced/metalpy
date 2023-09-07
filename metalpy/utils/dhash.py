import itertools
import os
import warnings

import cloudpickle
import numpy as np

from metalpy.utils.type import get_or_default, undefined


def dhash(*objs):
    """确定性哈希，通过将Python原生hash中非确定性的部分替换来实现，例如str，并通过扩展方法支持更多对象

    Parameters
    ----------
    objs
        待哈希的对象

    Returns
    -------
    ret : DHash
        DHash实例
    """
    return DHash(*objs)


def register_dhasher(*hashed_types):
    def wrapper(func):
        for t in hashed_types:
            DHash.hashers[t] = func
        return func
    return wrapper


def register_lazy_dhasher(hashed_type: str):
    def wrapper(func):
        DHash.hasher_creators[hashed_type] = func
        return func
    return wrapper


class DHash:
    hashers = {}
    hasher_creators = {}

    def __init__(self, *objs, convert=True):
        """确定性哈希，通过将Python原生hash中非确定性的部分替换来实现，例如str，并通过扩展方法支持更多对象

        Parameters
        ----------
        objs
            待哈希的对象

        Warnings
        --------
            DHash.__init__本身不会对输入的对象进行任何转换，默认均为dhash可接受对象
            请优先使用dhash()函数
        """
        if convert:
            self.objs = tuple(DHash.convert_to_dhashable(obj) for obj in objs)
        else:
            self.objs = objs

        self._result = None

    @property
    def result(self):
        if self._result is None:
            self._result = hash(self.objs)
        return self._result

    def __iter__(self):
        for obj in self.objs:
            if isinstance(obj, DHash):
                for o in obj:
                    yield o
            else:
                yield obj

    def digest(self):
        # TODO: 使用更合理的哈希函数，Python自带的哈希函数会导致结果存在负数
        return abs(self.result)

    def hexdigest(self, digits=32):
        hash_value = self.digest()
        char_table = '0123456789abcdef'
        tablesize = len(char_table)
        hash_string = ''

        while hash_value != 0 and len(hash_string) < digits:
            c = hash_value % tablesize
            hash_string = char_table[c] + hash_string
            hash_value = hash_value // tablesize

        return hash_string

    def __str__(self):
        return self.hexdigest(6)

    def __dhash__(self):
        return self

    def __hash__(self):
        return self.digest()

    def __eq__(self, other):
        return isinstance(other, DHash) and self.result == other.result

    @staticmethod
    def convert_to_dhashable(obj):
        t = type(obj)
        hasher = get_or_default(DHash.hashers, t, None)

        if hasher == undefined:
            hasher = None  # 已确认不存在lazy hasher
        else:
            if hasher is None:
                hasher = DHash._find_lazy_hasher(t)

        if hasher is None:
            def fallback(x): DHash(x, convert=False)
            hasher = getattr(t, '__dhash__', fallback)
            if hasher == fallback:
                warnings.warn(f'Cannot find dhasher for type `{t.__name__}`,'
                              f' falling back to built-in `hash`.')

        return hasher(obj)

    @staticmethod
    def _find_lazy_hasher(t):
        from metalpy.utils.object_path import ObjectPath
        type_name = str(ObjectPath.of(t))
        hasher = get_or_default(DHash.hasher_creators, type_name, None)

        if hasher is not None:
            hasher = DHash.hashers[t] = hasher
        else:
            DHash.hashers[t] = undefined

        return hasher


@register_dhasher(*itertools.chain(*[np.sctypes[k] for k in np.sctypes if k != 'others']))
@register_dhasher(float, int, bool)
def _hash_basic_type(obj):
    return DHash(obj, convert=False)


@register_dhasher(list)
def _hash_list(obj: list):
    return dhash(*obj)


@register_dhasher(tuple)
def _hash_list(obj: tuple):
    return dhash(*obj)


@register_dhasher(str)
def _hash_str(obj: str):
    from functools import reduce
    with np.errstate(over='ignore'):  # 消除上溢警告
        order = np.uint64(5)  # 解决uint64左移运算符的类型错误 https://github.com/numpy/numpy/issues/2524
        ret = reduce(lambda h, c: np.uint64(ord(c)) + ((h << order) + h), obj, np.uint64(5381))
    return ret


@register_dhasher(bytes)
def _hash_bytes(obj: bytes):
    buf = np.frombuffer(obj, dtype=np.uint8)

    from functools import reduce
    with np.errstate(over='ignore'):
        order = np.uint64(5)
        ret = reduce(lambda h, c: c + ((h << order) + h), buf, np.uint64(5381))
    return ret


@register_dhasher(type(None))
def _hash_none(_: None):
    return 0o0225


@register_dhasher(dict)
def _hash_dict(obj: dict):
    return dhash(*sorted(obj.items(), key=lambda x: x[0]))


@register_dhasher(type)
@register_dhasher(type(_hash_dict))  # function
def _hash_serializable(obj):
    return dhash(cloudpickle.dumps(obj))


@register_dhasher(np.ndarray)
def _hash_array(arr: np.ndarray, n_samples=10, sparse=False):
    if not sparse:
        arr = arr.ravel()
        rand = np.random.RandomState(int(arr[len(arr) // 2]) % 2**32)
        n_samples = min(n_samples, len(arr))

        # 加入shape作为参数防止类似np.ones(100)和np.ones(1000)的冲突
        return dhash((arr.shape, *rand.choice(arr, min(len(arr), n_samples), replace=False),))
    else:
        # 稀疏的数组可能会在采样时碰撞概率较大
        # 但是涉及压缩操作可能会导致效率一定程度上降低
        import blosc2
        packed = blosc2.pack_array2(arr)

        return dhash(packed)


@register_lazy_dhasher('pathlib:WindowsPath')
@register_lazy_dhasher('pathlib:PosixPath')
def _hash_path(path):
    return _hash_str(os.fspath(path))


@register_lazy_dhasher('pandas.core.series:Series')
def _hash_series(arr):
    return _hash_array(arr)
