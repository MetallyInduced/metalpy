import warnings

import numpy as np


class Coordinates(np.ndarray):
    def __init__(self, shape):
        super().__init__(shape)
        self.crs = None

    def __new__(cls, *args, crs=None, **kwargs):
        obj = super(Coordinates, cls).__new__(cls, *args, **kwargs)
        obj._init(crs)
        return obj

    def __array_finalize__(self, obj, **kwargs):
        self._init(getattr(obj, 'crs', None))

    def _init(self, crs=None):
        if crs is None:
            crs = 'WGS 84'

        self.crs = crs
        assert self.shape[-1] in (2, 3), 'Coordinates supports only as 2d or 3d array.'

    @staticmethod
    def warn_invalid_modification():
        warnings.warn('Modifying dims of Coordinates is not allowed.')

    def reshape(self, *_, **__):
        Coordinates.warn_invalid_modification()

    def with_crs(self, crs):
        self.crs = crs

    def warp(self, crs):
        """将所有坐标转换到指定的另一坐标系下

        Returns
        -------
        ret
            新坐标系下的坐标集合
        """
        pass
