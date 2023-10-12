from functools import cached_property, reduce

from metalpy.utils.bounds import union, intersects
from metalpy.utils.dhash import dhash
from . import Shape3D
from ..mix_modes import Mixer, MixMode, dhashable_mixer


class Composition(Shape3D):
    # Composition中只考虑正数意义，因此Max和Min分别为并和交
    Union = MixMode.Max
    Intersects = MixMode.Min

    def __init__(self, *shapes: Shape3D, mix_mode: Mixer = Union):
        super().__init__()
        if len(shapes) > 0:
            self.shapes = list(shapes)
        self.mix_mode = mix_mode

    @property
    def mixer(self):
        return MixMode.dispatch(self.mix_mode)

    @cached_property
    def shapes(self) -> list[Shape3D]:
        return []

    @property
    def n_tasks(self):
        return sum((s.n_tasks for s in self.shapes))

    @property
    def progress_manually(self):
        return True

    def do_place(self, mesh_cell_centers, progress):
        mixer = self.mixer
        ret = None
        for shape in self.shapes:
            indices = shape.place(mesh_cell_centers, progress)
            if ret is None:
                ret = indices
            else:
                ret = mixer(ret, indices)

            if progress is not None and not shape.progress_manually:
                progress.update(shape.n_tasks)

        return ret

    def do_compute_implicit_distance(self, mesh_cell_centers, progress):
        if self.mix_mode == Composition.Intersects:
            mixer = MixMode.max
        else:
            mixer = MixMode.min

        ret = None
        for shape in self.shapes:
            distances = shape.compute_implicit_distance(mesh_cell_centers, progress)
            if ret is None:
                ret = distances
            else:
                ret = mixer(ret, distances)

            if progress is not None and not shape.progress_manually:
                progress.update(shape.n_tasks)

        return ret

    def __len__(self):
        return len(self.shapes)

    def __iter__(self):
        yield from self.shapes

    def __getitem__(self, item):
        return self.shapes[item]

    def __dhash__(self):
        return dhash(super().__dhash__(), *self.shapes, dhashable_mixer(self.mix_mode))

    def do_clone(self, deep=True):
        return Composition(
            *[s.clone(deep=deep) for s in self.shapes],
            mix_mode=self.mix_mode
        )

    @property
    def local_bounds(self):
        if self.mix_mode == Composition.Intersects:
            reducer = intersects
        else:
            reducer = union
        return reduce(reducer, (shape.bounds for shape in self))

    def to_local_polydata(self):
        # TODO: 基于模型布尔操作实现常见混合模式下的对象合并处理
        import pyvista as pv
        shapes = [shape.to_polydata() for shape in self]
        shape = pv.MultiBlock(shapes)

        return shape
