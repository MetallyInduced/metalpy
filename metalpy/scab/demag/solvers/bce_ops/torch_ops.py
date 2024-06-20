from typing import Iterable

import torch

from metalpy.utils.taichi import ti_size_dtype
from . import BCEOps
from ..bce import cast_numpy_dtype_to_torch


class TorchOps(BCEOps):
    def from_array(self, arr):
        import torch
        return torch.from_numpy(arr).to(self.device)

    def array(self, shape, dtype=None, fill=0.):
        if dtype is None:
            dtype = self.kernel_dtype

        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            shape = (shape,)

        # 当前torch和taichi均支持的架构只有 `cuda` 或 `cpu`
        return torch.full(
            shape,
            fill,
            dtype=cast_numpy_dtype_to_torch(dtype),
            device=self.device
        )

    def copy(self, arr):
        return arr.clone()

    def arange(self, size):
        return torch.arange(size, dtype=cast_numpy_dtype_to_torch(ti_size_dtype), device=self.device)

    def where(self, arr):
        return torch.where(arr)

    def flatten(self, arr):
        return arr.flatten()

    def rfftn(self, arr, out=None, s=None, real=False, imag=False):
        if s is not None:
            s = tuple(s)

        from torch.fft import rfftn
        rfftn(arr, s=s, out=self.tmp_complex)
        ret = self.tmp_complex

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

        from torch.fft import irfftn
        irfftn(arr, s=s, out=self.tmp)
        ret = self.tmp

        if out is not None:
            out[:] = ret
            return out
        else:
            return ret

    def iadd(self, a, b, alpha=1):
        torch.add(a, b, alpha=alpha, out=a)

    def reciprocal(self, a):
        torch.reciprocal(a, out=a)
        return a
