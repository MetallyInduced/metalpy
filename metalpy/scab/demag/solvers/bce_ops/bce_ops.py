from ..bce import cast_float_to_complex


class BCEOps:
    def __init__(self, kernel_dtype, device, real_tmp_shape, complex_tmp_shape):
        self.kernel_dtype = kernel_dtype
        self.device = device
        self.tmp = self.array(real_tmp_shape)
        self.tmp_complex = self.array(complex_tmp_shape, dtype=self.complex_kernel_dtype)

    @property
    def complex_kernel_dtype(self):
        return cast_float_to_complex(self.kernel_dtype)

    def from_array(self, arr):
        raise NotImplementedError

    def array(self, shape, dtype=None, fill=0.):
        raise NotImplementedError

    def array_like(self, arr, dtype=None):
        if dtype is None:
            dtype = arr.dtype
        return self.array(arr.shape, dtype=dtype)

    def copy(self, arr):
        raise NotImplementedError

    def arange(self, size):
        raise NotImplementedError

    def where(self, arr):
        raise NotImplementedError

    def flatten(self, arr):
        raise NotImplementedError

    def rfftn(self, arr, out=None, s=None, real=False, imag=False):
        raise NotImplementedError

    def irfftn(self, arr, out=None, s=None):
        raise NotImplementedError

    def iadd(self, a, b, alpha=1):
        raise NotImplementedError

    def reciprocal(self, a):
        raise NotImplementedError
