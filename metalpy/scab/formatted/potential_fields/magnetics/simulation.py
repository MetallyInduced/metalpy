from metalpy.mexin import mixin
from ..base import BasePFDataFormatterMixin


class FormattedSimulation3DIntegralMixin(BasePFDataFormatterMixin):
    def format_numpy_data(self, this, data, components):
        return data.reshape(-1, len(components))


class FormattedSimulation3DDifferentialMixin(BasePFDataFormatterMixin):
    def format_numpy_data(self, this, data, components):
        return data.reshape(len(components), -1).T

    @mixin.after(keep_retval='ret')
    def projectFields(self, *_, ret, **__):
        return self.format_data(ret)
