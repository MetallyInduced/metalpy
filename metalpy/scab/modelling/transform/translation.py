import numpy as np

from metalpy.utils.dhash import dhash
from .transform import Transform


class Translation(Transform):
    def __init__(self, x, y, z):
        super().__init__()
        self.delta = np.asarray([x, y, z])

    def transform(self, mesh):
        return np.asarray(mesh) + self.delta

    def inverse_transform(self, mesh):
        return np.asarray(mesh) - self.delta

    def clone(self):
        return Translation(*self.delta)

    def __dhash__(self):
        return dhash(*self.delta)
