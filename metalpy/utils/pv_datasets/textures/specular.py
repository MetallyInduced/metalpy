import numpy as np

from . import Texture, EmptyTexture


class Specular(Texture):
    def __init__(self, color, shininess):
        super().__init__(np.r_[color, shininess])

    @classmethod
    def extract(cls, dataset, name=None):
        data = cls.extract_field(dataset, name=name)
        if data is None:
            return EmptyTexture()
        return cls(data[:3], data[-1])

    @property
    def color(self):
        return self.texture[:3]

    @property
    def shininess(self):
        return self.texture[-1]

    def apply(self, actor):
        actor.prop.specular_color = self.color
        # PyVista的默认亮度为100.0，但mtl默认为0.0
        actor.prop.specular_power = min(self.shininess + 100, 128)
        actor.prop.specular = 1
