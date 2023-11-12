from __future__ import annotations

import abc
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pyvista as pv

from metalpy.utils.file import openable, PathLike
from metalpy.utils.type import get_or_default
from .textures import TextureSet, Diffuse, Ambient, Specular, Alpha


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


class DictWithFallback:
    def __init__(self, fallbacks):
        self.dict = {}
        self.fallbacks = fallbacks

    def __setitem__(self, item, value):
        self.dict[item] = value

    def __getitem__(self, item):
        return get_or_default(self.dict, item, supplier=lambda: self.fallbacks[item])

    def get(self, *args, fallback=False, **kwargs):
        if fallback:
            kwargs['supplier'] = lambda: self.fallbacks.get(*args, **kwargs)
        return get_or_default(self.dict, *args, **kwargs)


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

        texture_mappings = {}
        for mtl_path in mtl_paths:
            if not mtl_path.exists():
                return obj_mesh

            # parse the mtl file
            for mtl_name, material in ObjTextureReader.parse_mtl(mtl_path):
                textures = TextureSet()
                texture_mappings[mtl_name] = textures

                illum = material['illum']

                if illum >= 0:
                    # Color on
                    kd = material.get('Kd', None)
                    if kd is not None:
                        if isinstance(kd, float):
                            kd = [kd] * 3
                        textures.append(Diffuse(kd))

                if illum >= 1:
                    # Ambient on
                    ka = material.get('Ka', None)
                    if ka is not None:
                        if isinstance(ka, float):
                            ka = [ka] * 3
                        textures.append(Ambient(ka))

                if illum >= 2:
                    # Highlight on
                    ks = material.get('Ks', None)
                    if ks is not None:
                        if isinstance(ks, float):
                            ks = [ks] * 3
                        textures.append(Specular(ks, material['Ns']))

                d = material.get('d', None)
                if d is not None:
                    textures.append(Alpha(d))

                tr = material.get('Tr', None)
                if tr is not None:
                    textures.append(Alpha(1 - tr))

        n_textured = len(texture_mappings)

        if n_textured < 1:
            model = obj_mesh
        elif n_textured == 1:
            model = obj_mesh
            for name, tex in texture_mappings.items():
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
                materials = list(texture_mappings.keys())
                materials.sort()

            for i in np.unique(material_ids):
                name = materials[i]
                mesh_part = obj_mesh.extract_cells(material_ids == i)
                texture_mappings[name].bind(mesh_part, name=name)
                model[name] = mesh_part

        return model

    MaterialBase = {
        'Ka': [0.0, 0.0, 0.0, 1.0],  # ambient
        'Ke': [0.0, 0.0, 0.0, 1.0],  # emissive
        'Kd': [0.5, 0.5, 0.5, 1.0],  # diffuse
        'Ks': [0.0, 0.0, 0.0, 1.0],  # specular
        'Ns': 0.0,  # specular shininess
        'd': 1.0,  # opacity (d / Tr)
        'illum': 2,  # illumination mode
    }

    @staticmethod
    def make_default_material():
        return DictWithFallback(ObjTextureReader.MaterialBase)

    @staticmethod
    def parse_mtl(mtl_path):
        cwd = mtl_path.parent
        with open(mtl_path) as mtl_file:
            material = None
            mtl_name = None
            for line in mtl_file.readlines():
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                slot = parts[0]
                params = parts[1:]
                if slot.startswith('#'):
                    pass
                elif slot == 'newmtl':
                    if material is not None:
                        yield mtl_name, material
                    material = ObjTextureReader.make_default_material()
                    mtl_name = params[0]
                else:
                    if slot.startswith('map'):
                        path = params[-1]  # 跳过选项
                        material[slot.split('_')[1]] = cwd / path
                    else:
                        fmt = params[0]
                        if fmt in ['spectral', 'xyz']:
                            warnings.warn(f'Format `spectral` and `xyz` are not supported.')
                        else:
                            rgb = [float(c) for c in params]
                            if len(rgb) == 1:
                                rgb = rgb[0]
                            material[slot] = rgb

            if material is not None:
                yield mtl_name, material
