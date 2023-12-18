from scipy.spatial.transform import Rotation as R

from metalpy.utils.dhash import dhash
from metalpy.utils.string import parse_axes_labels
from .transform import Transform


class Rotation(Transform):
    def __init__(self, a, b, y, degrees=True, seq='xyz', radians=False):
        super().__init__()

        if radians:
            degrees = False

        self.params = (a, b, y, degrees, seq)

        angles = [a, b, y]
        axes = parse_axes_labels(seq.lower())
        angles = [angles[axis] for axis in axes]

        # TODO: 解决Rotation精度问题
        self.rot = R.from_euler(seq, angles, degrees=degrees)

    def transform(self, mesh):
        return self.rot.apply(mesh)

    def inverse_transform(self, mesh):
        return self.rot.apply(mesh, inverse=True)

    def clone(self):
        return Rotation(*self.params)

    def __dhash__(self):
        return dhash(*self.params[:-1], self.params[-1])
