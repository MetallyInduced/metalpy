from __future__ import annotations

import warnings

from SimPEG.simulation import BaseSimulation

from metalpy.mexin import Patch, Mixin
from metalpy.mexin.utils import TypeMap
from metalpy.scab.distributed.policies import NotDistributable
from metalpy.utils.object_path import get_full_qualified_path


class FormattedContext(Mixin):
    _implementations = TypeMap()

    def __init__(self, this, pandas=False, locations=False):
        super().__init__(this)
        self.pandas = pandas
        self.locations = locations

    def post_apply(self, this):
        impl = FormattedContext._implementations.get(type(this))

        if impl is None:
            warnings.warn(
                f'Formatter support for {get_full_qualified_path(type(this))} is not implemented.'
                f' Ignoring it.'
            )
            return

        this.mixins.add(impl, pandas=self.pandas, locations=self.locations)


class Formatted(Patch, NotDistributable):
    def __init__(self, pandas=False, locations=False):
        """格式化SimPEG正演类的预测结果，将结果从展平的状态转换为二维矩阵

        Parameters
        ----------
        pandas
            如果为True，则格式化为 `pandas.DataFrame` ，行名为数据名（例如 'tmi' ， 'bx' 等）
        locations
            如果为True，则在格式化结果的左侧附加观测点的坐标列
        """
        super().__init__()
        self.pandas = pandas
        self.locations = locations

    def apply(self):
        self.add_mixin(BaseSimulation, FormattedContext, pandas=self.pandas, locations=self.locations)


def __implements(target):
    def decorator(func):
        FormattedContext._implementations.map(target, func)
        return func
    return decorator


@__implements('SimPEG.potential_fields.magnetics.simulation.Simulation3DIntegral')
def _():
    from .potential_fields.magnetics.simulation import FormattedSimulation3DIntegralMixin
    return FormattedSimulation3DIntegralMixin


@__implements('SimPEG.potential_fields.magnetics.simulation.Simulation3DDifferential')
def _():
    from .potential_fields.magnetics.simulation import FormattedSimulation3DDifferentialMixin
    return FormattedSimulation3DDifferentialMixin
