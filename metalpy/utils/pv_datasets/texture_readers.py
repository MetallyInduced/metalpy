from __future__ import annotations

import abc
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pyvista as pv

from metalpy.utils.file import openable, PathLike
from .textures import Diffuse


class TextureHelper:
    def __init__(self):
        self.readers: dict[str, TextureReader] = {}

    def get_reader(self, fmt):
        reader = self.readers.get(fmt, None)
        if reader is None:
            reader = TextureReader.find(fmt)()
            self.readers[fmt] = reader

        return reader

    def bind_texture(self, model, model_path):
        model_path = Path(model_path)
        return self.get_reader(model_path.suffix).bind_texture(model, cwd=model_path.parent)


class TextureReader(abc.ABC):
    _Readers: dict[str, type[TextureReader]] = {}

    @abc.abstractmethod
    def bind_texture(self, model, cwd):
        pass

    @staticmethod
    def of(*formats):
        def wrapper(cls):
            for fmt in formats:
                TextureReader._Readers[fmt] = cls

            return cls

        return wrapper

    @staticmethod
    def find(fmt) -> type[TextureReader]:
        ret = TextureReader._Readers.get(fmt, None)
        if ret is None:
            raise RuntimeError(f'Reading texture for `{fmt}` is currently not supported.')
        else:
            return ret


@TextureReader.of('.obj')
class ObjTextureReader(TextureReader):
    def bind_texture(self, model, cwd):
        return ObjTextureReader.bind_mtl_textures(obj_mesh=model, cwd=cwd)

    @staticmethod
    def bind_mtl_textures(obj_mesh, mtl_path: PathLike | Iterable[PathLike] = None, cwd=None):
        """Modified from @109021017's comment:
        https://github.com/pyvista/pyvista-support/issues/514#issuecomment-1021260384
        """
        if openable(mtl_path):
            mtl_paths = [Path(mtl_path)]
        elif mtl_path is None:
            mtl_paths = [Path(cwd) / p for p in obj_mesh.field_data['MaterialLibraries']]
        else:
            mtl_paths = [Path(p) for p in mtl_path]

        n_textured = 0
        texture_paths = {}
        for mtl_path in mtl_paths:
            texture_dir = mtl_path.parent
            mtl_name = None

            if not mtl_path.exists():
                return obj_mesh

            # parse the mtl file
            with open(mtl_path) as mtl_file:
                for line in mtl_file.readlines():
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    if parts[0] == 'map_Kd':
                        if mtl_name is not None:
                            texture_paths[mtl_name] = Diffuse(texture_path=texture_dir / parts[1])
                            n_textured += 1
                    elif parts[0] == 'Kd':
                        if mtl_name is not None:
                            texture_paths[mtl_name] = Diffuse(texture=[float(c) for c in parts[1:]])
                            n_textured += 1
                    elif parts[0] == 'newmtl':
                        mtl_name = parts[1]
                        texture_paths[mtl_name] = None

        if n_textured < 1:
            model = obj_mesh
        elif n_textured == 1:
            model = obj_mesh
            for name, tex in texture_paths.items():
                if tex is not None:
                    tex.bind(model, name=name)
                    break
        else:
            material_ids = obj_mesh.cell_data['MaterialIds']
            model = pv.MultiBlock()

            materials = obj_mesh.field_data.get('MaterialNames', None)
            if materials is None:
                materials = obj_mesh.cell_data.get('Materials', None)
            if materials is None:
                warnings.warn('No materials list found.'
                              ' Using texture names, which may lead to unexpected result.')
                materials = list(texture_paths.keys())
                materials.sort()

            for i in np.unique(material_ids):
                name = materials[i]
                mesh_part = obj_mesh.extract_cells(material_ids == i)
                texture_paths[name].bind(mesh_part, name=name)
                model[name] = mesh_part

        return model
