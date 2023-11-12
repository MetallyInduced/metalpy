from . import Coefficient


class Alpha(Coefficient):
    @property
    def name(self):
        return 'opacity'

    def build(self):
        if self.texture == 1:
            return {}
        else:
            return super().build()
