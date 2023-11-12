from . import Coefficient


class PBR(Coefficient):
    def __init__(self, value):
        """目前 VTK 中的PBR相关参数都是以系数形式提供，因此直接继承自 `Coefficient`

        TODO: 将来如果有新特性可能需要跟进
        """
        super().__init__(value)

    def build(self):
        return {
            'pbr': True,
            **super().build()
        }


class Metallic(PBR):
    pass


class Roughness(PBR):
    pass
