import numpy as np

from metalpy.utils.bounds import Bounds
from metalpy.utils.dhash import dhash
from . import Shape3D


class FullSpace(Shape3D):
    def __init__(self):
        """定义一个覆盖整个空间的几何体
        """
        super().__init__()

    def do_place(self, mesh_cell_centers, progress):
        indices = np.ones(mesh_cell_centers.shape[0], dtype=bool)
        return indices

    def do_hash(self):
        return 0

    def __dhash__(self):
        return dhash()

    def do_clone(self, deep=True):
        return FullSpace()

    @property
    def local_bounds(self):
        return Bounds.unbounded()

    def to_local_polydata(self):
        return None
