import numpy as np


class SharedArray(np.ndarray):
    def __new__(cls, *args, shared_memory, **kwargs):
        ret: SharedArray = np.ndarray(*args, **kwargs, buffer=shared_memory.buf).view(SharedArray)
        ret._init(shared_memory)

        return ret

    def __array_finalize__(self, obj, **kwargs):
        self._init(getattr(obj, 'shm', None))

    def __reduce_ex__(self, protocol):
        return self._build, (self.shape, self.dtype, self.shm)

    def _init(self, shm):
        self.shm = shm

    @classmethod
    def _build(cls, shape, dtype, shm):
        return cls(shape=shape, dtype=dtype, shared_memory=shm)
