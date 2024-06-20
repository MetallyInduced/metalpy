from typing import Iterable

import numpy as np
from scipy.fft import rfftn, irfftn

from metalpy.utils.taichi import ti_size_dtype
from . import BCEOps


class NumpyOps(BCEOps):
    def from_array(self, arr):
        return np.asarray(arr)

    def array(self, shape, dtype=None, fill=0.):
        if dtype is None:
            dtype = self.kernel_dtype

        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            shape = (shape,)

        return np.full(shape, fill, dtype=dtype)

    def copy(self, arr):
        return arr.copy()

    def arange(self, size):
        return np.arange(size, dtype=ti_size_dtype)

    def where(self, arr):
        return np.where(arr)

    def flatten(self, arr):
        return arr.ravel()

    def rfftn(self, arr, out=None, s=None, real=False, imag=False):
        if s is not None:
            s = tuple(s)

        # TODO: 改用in-place fft实现？
        ret = rfftn(arr, s=s, overwrite_x=True)

        if real:
            ret = ret.real

        if imag:
            ret = ret.imag

        if out is not None:
            out[:] = ret
            return out
        else:
            return ret

    def irfftn(self, arr, out=None, s=None):
        if s is not None:
            s = tuple(s)

        ret = irfftn(arr, s=s, overwrite_x=True)

        if out is not None:
            out[:] = ret
            return out
        else:
            return ret

    def iadd(self, a, b, alpha=1):
        a[:] += alpha * b

    def reciprocal(self, a):
        a[:] = 1 / a
        return a
