import pyvista as pv

from .texture import Texture


class Diffuse(Texture):
    def __init__(self, texture=None, texture_path=None):
        super().__init__(texture, texture_path)

    def build(self):
        tex = self.texture
        if tex.shape == (3,):
            return {'color': tex}
        else:
            return {'texture': pv.Texture(tex)}
