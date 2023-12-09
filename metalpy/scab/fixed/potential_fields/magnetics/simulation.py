import types

from SimPEG.potential_fields.magnetics import Simulation3DDifferential

from metalpy.mexin import Mixin


class FixedSimulation3DDifferentialMixin(Mixin):
    def __init__(self, this, dpred, **_):
        super().__init__(this)

        if dpred:
            # 选择性开启
            self.dpred = types.MethodType(FixedSimulation3DDifferentialMixin.dpred, self)

    @staticmethod
    def dpred(self, this: Simulation3DDifferential, m, f=None):
        assert f is None
        u = this.fields(m)
        return this.projectFields(u)
