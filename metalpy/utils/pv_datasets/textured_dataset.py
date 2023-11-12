from __future__ import annotations

import pyvista as pv

from metalpy.mepa.executor import Executor
from metalpy.utils.model import pv_ufunc_apply
from .dataset_wrapper import TDataSetWrapper
from .textures import Texture
from .texture_readers import TextureHelper
from .universal_dataset import UniversalDataSet


class TexturedDataSet(UniversalDataSet):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

    @classmethod
    def read(
            cls: type[TDataSetWrapper],
            model_path,
            texture=True,
            helper: TextureHelper = None
    ) -> TDataSetWrapper | pv.DataSet:
        if texture:
            helper_func = TextureHelper().bind_texture
        else:
            helper_func = None

        ret = super().read(model_path, callback=helper_func)

        return ret.view(TexturedDataSet)

    @classmethod
    def read_all(
            cls: type[TDataSetWrapper],
            path,
            progress=True,
            verbose=True,
            cache=True,
            preview: int | bool = False,
            texture=True,
            executor: Executor | None = None
    ) -> TDataSetWrapper | pv.DataSet:
        if texture:
            helper_func = TextureHelper().bind_texture
        else:
            helper_func = None

        return super().read_all(
            path,
            progress=progress,
            verbose=verbose,
            cache=cache,
            preview=preview,
            executor=executor,
            callback=helper_func,
        ).view(TexturedDataSet)

    def plot(self, texture=True, **kwargs):
        plotter = pv.Plotter()

        show_grid = kwargs.pop('show_grid', False)
        if show_grid:
            plotter.show_grid()

        show_axes = kwargs.pop('show_axes', False)
        if show_axes:
            plotter.show_axes()

        self.add_to_plotter(plotter, texture=texture, **kwargs)

        plotter.show()

    def add_to_plotter(self, plotter: pv.BasePlotter, *, texture=True, **kwargs):
        if texture:
            pv_ufunc_apply(self.dataset, lambda dataset: _add_textured_dataset_to_plotter(
                dataset,
                plotter,
                **kwargs
            ))
        else:
            plotter.add_mesh(self.dataset, **kwargs)


def _add_textured_dataset_to_plotter(dataset, plotter: pv.BasePlotter, **kwargs):
    if dataset.n_points <= 0:
        return

    textures = Texture.extract_all(dataset)
    new_kwargs = textures.build()
    new_kwargs.update(kwargs)

    actor = plotter.add_mesh(dataset, **new_kwargs)
    textures.apply(actor)
