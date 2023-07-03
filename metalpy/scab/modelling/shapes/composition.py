from functools import cached_property

from metalpy.utils.dhash import dhash
from . import Shape3D
from .shape3d import bounding_box_of
from ..mix_modes import Mixer, MixMode, dhashable_mixer


class Composition(Shape3D):
    def __init__(self, *shapes: Shape3D, mix_mode: Mixer = MixMode.Max):
        super().__init__()
        if len(shapes) > 0:
            self.shapes = list(shapes)
        self.mix_mode = mix_mode

    @property
    def mixer(self):
        return MixMode.dispatch(self.mix_mode)

    @cached_property
    def shapes(self):
        return []
    
    def do_place(self, mesh_cell_centers, worker_id):
        ret = None
        for shape in self.shapes:
            indices = shape.place(mesh_cell_centers, worker_id)
            if ret is None:
                ret = indices
            else:
                ret = self.mixer(ret, indices)

        return ret

    def __len__(self):
        return len(self.shapes)

    def __iter__(self):
        yield from self.shapes

    def __getitem__(self, item):
        return self.shapes[item]

    def do_hash(self):
        return hash((*self.shapes, dhash(dhashable_mixer(self.mix_mode))))

    def __dhash__(self):
        return dhash(super().__dhash__(), *self.shapes, dhashable_mixer(self.mix_mode))

    def do_clone(self):
        return Composition(*self.shapes, mix_mode=self.mix_mode)

    @property
    def local_bounds(self):
        return bounding_box_of(self)

    def to_local_polydata(self):
        import pyvista as pv
        ret = pv.MultiBlock()
        for shape in self:
            ret.append(shape.to_polydata())

        return ret
