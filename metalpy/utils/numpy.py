from typing import Mapping, Iterable, Sequence

import numpy as np


def ensure_as_numpy_array_collection(array_collection):
    if isinstance(array_collection, dict):
        return {k: np.asarray(v) for k, v in array_collection.items()}
    elif isinstance(array_collection, list):
        return [np.asarray(v) for v in array_collection]
    elif isinstance(array_collection, tuple):
        return (np.asarray(v) for v in array_collection)
    elif isinstance(array_collection, Mapping):
        return {k: np.asarray(v) for k, v in array_collection.items()}
    elif isinstance(array_collection, Iterable):
        return [np.asarray(v) for v in array_collection]


def array_homogeneous_key(arr: np.ndarray):
    return arr.shape, arr.dtype


def is_homogeneous(inp: np.ndarray, out: np.ndarray):
    if inp.shape != out.shape:
        # 不允许数组尺寸发生改变
        return False
    if out.dtype != np.promote_types(inp.dtype, out.dtype):
        # 输出只允许类型抬升
        return False

    return True


class FixedShapeNDArray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(*args, **kwargs)

    def __array_finalize__(self, _, **__):
        pass

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        typ = type(self)

        inputs = tuple(np.asarray(inp) if isinstance(inp, typ) else inp for inp in inputs)
        if out is not None:
            out = tuple(np.asarray(o) if isinstance(o, typ) else o for o in out)
        ret = super().__array_ufunc__(ufunc, method, *inputs, out=out, **kwargs)

        if ret is NotImplemented:
            return NotImplemented

        if is_homogeneous(self, ret):
            return ret.view(typ)
        else:
            return ret

    def reshape(
        self, *shape, **kwargs
    ):
        return self.view(np.ndarray).reshape(*shape, **kwargs)

    def __getitem__(self, item):
        return super().__getitem__(item).view(np.ndarray)


def is_dtype(dt):
    if isinstance(dt, np.dtype):
        return True

    if not isinstance(dt, type):
        return False

    if dt in (bool, int, float, complex, str):
        # Python内置类型作为numpy类型的别名
        # https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
        return True
    elif issubclass(dt, np.generic):
        # numpy 标量类型基类
        return True

    return False


def dtype_of(arr, default=None):
    dtype = None

    if is_dtype(arr):
        dtype = arr

    if dtype is None:
        dtype = getattr(arr, 'dtype', None)  # 如果已经是数组类型，则获取 dtype 属性

    if dtype is None:
        # 代表不是数组类型
        if isinstance(arr, Sequence):
            if len(arr) > 0:
                dtype = dtype_of(arr[0])
            else:
                dtype = default
        else:
            dtype = np.asarray(arr).dtype

    if dtype is None:
        raise RuntimeError('Unable to detect the dtype of input array.')

    return dtype


def get_resolution(arr):
    dtype = dtype_of(arr, default=float)
    if np.issubdtype(dtype, np.integer):
        dtype = float

    return np.finfo(dtype).resolution * 10
