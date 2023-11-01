import abc
import warnings
from pathlib import Path

import imageio
import numpy as np

from metalpy.utils.model import DataSetLike


class Texture(abc.ABC):
    def __init__(self, texture=None, texture_path=None):
        if texture is not None:
            texture = np.asarray(texture)
        else:
            assert texture_path is not None, 'Either one of `texture` and `texture_path` must be provided.'

        self._texture = texture
        self.texture_path = texture_path

    @abc.abstractmethod
    def build(self):
        pass

    @property
    def texture(self):
        if self._texture is None:
            self._texture = imageio.v3.imread(self.texture_path)
        return self._texture

    @texture.setter
    def texture(self, tex):
        self._texture = tex

    @property
    def is_map(self):
        return self.texture.ndim > 1

    def bind(self, dataset: DataSetLike, name=None):
        if name is None:
            if self.texture_path is not None:
                name = Path(self.texture_path).stem

        tex = self.texture
        tex_name, shape_name = self.field_name(name)
        dataset.field_data[tex_name] = tex.ravel()
        dataset.field_data[shape_name] = tex.shape

        if self.is_map:
            # 贴图需要额外激活 t_coords
            if name is not None:
                dataset.point_data.active_t_coords_name = name
            else:
                warnings.warn('Unnamed texture encountered, failed to bind t_coord.')

    @staticmethod
    def extract_all(dataset, name=None):
        textures = TextureSet()
        for texture_type in Texture.__subclasses__():
            tex = texture_type.extract(dataset, name=name)
            if not isinstance(tex, EmptyTexture):
                textures.append(tex)

        return textures

    @classmethod
    def extract(cls, dataset, name=None):
        tex = None
        if name is None:
            for tex_name in dataset.field_data:
                tex_name, shape_name = cls.check_field_name(tex_name)
                if tex_name is None:
                    continue
                tex = dataset.field_data.get(tex_name, None)
                if tex is not None:
                    shape = dataset.field_data[shape_name]
                    tex = tex.reshape(shape)
                    break
        else:
            tex_name, shape_name = cls.field_name(name)
            tex = dataset.field_data[tex_name]
            shape = dataset.field_data[shape_name]
            tex = tex.reshape(shape)

        if tex is None:
            return EmptyTexture()

        return cls(tex)

    @classmethod
    def field_name(cls, name):
        prefix = cls.__name__
        return f'{prefix}[{name}]', f'{prefix}Shape[{name}]'

    @classmethod
    def check_field_name(cls, tex_name: str):
        prefix = cls.__name__
        if tex_name.startswith(prefix):
            name = tex_name.split('[', maxsplit=1)[1][:-1]
            return f'{prefix}[{name}]', f'{prefix}Shape[{name}]'
        else:
            return None, None


class EmptyTexture(Texture):
    def __init__(self):
        super().__init__(texture=[0])

    def build(self):
        return {}

    @classmethod
    def extract(cls, dataset, name=None):
        return EmptyTexture()


class TextureSet(list[Texture]):
    def build(self):
        ret = {}
        for tex in self:
            ret.update(tex.build())

        return ret
