import warnings

import numpy as np

from metalpy.utils.dhash import dhash
from . import InfiniteShape


class FullSpace(InfiniteShape):
    def __init__(self):
        """定义一个覆盖整个空间的几何体
        """
        super().__init__()

    def do_place(self, mesh_cell_centers, progress):
        indices = np.ones(mesh_cell_centers.shape[0], dtype=bool)
        return indices

    def __dhash__(self):
        return dhash(0)

    def do_clone(self, deep=True):
        return FullSpace()

    def to_local_polydata(self):
        return None

    @property
    def volume(self):
        warnings.warn(f'Trying to access `volume` of `{FullSpace.__name__}`.')
        return 0

    @property
    def area(self):
        warnings.warn(f'Trying to access `area` of `{FullSpace.__name__}`.')
        return 0
