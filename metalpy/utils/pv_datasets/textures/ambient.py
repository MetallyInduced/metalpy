from . import Texture


class Ambient(Texture):
    def apply(self, actor):
        actor.prop.ambient_color = self.texture
        # 系数如果取1会过曝，难道是PyVista的plotter默认亮度太高了？
        actor.prop.ambient = 0.1
