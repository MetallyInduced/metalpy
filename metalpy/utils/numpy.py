import numpy as np


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

    def __getitem__(self, item):
        return super().__getitem__(item).view(np.ndarray)
