from __future__ import annotations

from typing import Iterable, overload

from SimPEG import maps
from SimPEG.potential_fields import magnetics

from metalpy.scab.utils.misc import define_inducing_field
from .. import BasePFSimulationBuilder
from ... import SimulationBuilder


class Simulation3DIntegralBuilder(BasePFSimulationBuilder):
    def __init__(self, sim_cls: magnetics.Simulation3DIntegral):
        super().__init__(sim_cls)
        self._receiver_list = []
        self._source_field = None
        self._model_type = None

        self.source_field(50000, 90, 0)
        self.scalar_model()

    def build(self) -> magnetics.Simulation3DIntegral:
        return super().build()

    def receivers(self, receiver_points, components: str | Iterable[str] = 'tmi'):
        self._receiver_list.append(
            magnetics.receivers.Point(
                receiver_points, components=components
            ))

    def source_field(self, strength, inc, dec):
        self._source_field = define_inducing_field(strength, inc, dec)

    @SimulationBuilder._assembles('survey')
    def _survey(self):
        field = magnetics.sources.UniformBackgroundField(
            receiver_list=self._receiver_list,
            amplitude=self._source_field.strength,
            inclination=self._source_field.inclination,
            declination=self._source_field.declination,
        )
        survey = magnetics.survey.Survey(field)

        return survey

    @overload
    def model_type(self, model_type: str): ...

    @overload
    def model_type(self, *, scalar: bool): ...

    @overload
    def model_type(self, *, vector: bool): ...

    @SimulationBuilder._supplies('model_type')
    def model_type(self, type=None, *, scalar=None, vector=None, **_):
        """设置模型类型，对应Simulation3DIntegral的model_type参数。

        `scalar`代表标量模型，输入为每个网格的磁化率。

        `vector`代表向量模型，输入为每个网格的等效磁化率向量，
        为(n_cells, 3)形式模型的按Fortran序展开（flatten(order='F')）
        或使用discretize.utils的mkvc。
        """

        assert (type is None) + (scalar is None) + (vector is None) == 2, \
            'Only one of `model_type`, `scalar` or `vector` can be provided for `model_type()`.'
        if type is not None:
            pass
        elif scalar is True:
            type = 'scalar'
        elif vector is True:
            type = 'vector'

        self._model_type = type
        return type

    def scalar_model(self):
        return self.model_type(scalar=True)

    def vector_model(self):
        return self.model_type(vector=True)

    @SimulationBuilder._assembles('chiMap')
    def _chi_map(self):
        n_params = self._model_mesh.n_active_cells
        if self._model_type == 'vector':
            n_params *= 3
        return maps.IdentityMap(nP=n_params)
