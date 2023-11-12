from .texture import Texture


class Coefficient(Texture):
    def __init__(self, value):
        super().__init__(value, texture_path=None)

    @property
    def name(self):
        return type(self).__name__.lower()

    def build(self):
        return {
            self.name: self.texture.item()
        }
