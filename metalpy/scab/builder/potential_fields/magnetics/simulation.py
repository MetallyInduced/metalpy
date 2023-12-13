from __future__ import annotations

from typing import Iterable, overload

import numpy as np
from SimPEG import maps
from SimPEG.potential_fields import magnetics

from metalpy.scab.potential_fields import magnetics as metalpy_magnetics
from metalpy.scab.utils.misc import define_inducing_field
from metalpy.utils.type import undefined, Dummy
from .. import BasePFSimulationBuilder
from ... import SimulationBuilder
from ...base import BaseMagneticPDESimulationBuilder


class MagneticSurveyBuilder(SimulationBuilder):
    def __init__(self, sim_cls, survey_cls=magnetics.Survey):
        super().__init__(sim_cls)
        self._survey_cls = survey_cls  # 继承类可以修改该属性以实现自定使用的Survey类

        self._receiver_list = []
        self._source_field = None

        self.source_field(50000, 90, 0)

    def receivers(self, receiver_points, components: str | Iterable[str] = 'tmi'):
        self._receiver_list.append(
            magnetics.receivers.Point(
                np.atleast_2d(receiver_points), components=components
            ))

    def source_field(self, strength, inc, dec):
        self._source_field = define_inducing_field(strength, inc, dec)

    @SimulationBuilder._assembles('survey')
    def _survey(self):
        if len(self._receiver_list) == 0:
            raise ValueError('No receivers defined. Please specify receivers with `builder.receivers()`.')

        field = magnetics.sources.UniformBackgroundField(
            receiver_list=self._receiver_list,
            amplitude=self._source_field.intensity,
            inclination=self._source_field.inclination,
            declination=self._source_field.declination,
        )
        survey = self._survey_cls(field)

        return survey


class Simulation3DIntegralBuilder(BasePFSimulationBuilder, MagneticSurveyBuilder):
    def __init__(self, sim_cls: type[magnetics.Simulation3DIntegral]):
        super().__init__(sim_cls)
        self._model_type = None

        self.scalar_model()

    def build(self) -> magnetics.Simulation3DIntegral:
        return super().build()

    @overload
    def model_type(self, model_type: str): ...

    @overload
    def model_type(self, *, scalar: bool): ...

    @overload
    def model_type(self, *, vector: bool): ...

    @SimulationBuilder._supplies('model_type')
    def model_type(self, type=None, *, scalar=None, vector=None):
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

    @SimulationBuilder._supplies('chiMap')
    def chi_map(self, chi_map: maps.IdentityMap | Dummy = undefined):
        return chi_map

    @SimulationBuilder._assembles('chiMap')
    def _chi_map(self):
        n_params = self.n_active_cells
        if self._model_type == 'vector':
            n_params *= 3
        return maps.IdentityMap(nP=n_params)


class Simulation3DDifferentialBuilder(BaseMagneticPDESimulationBuilder, MagneticSurveyBuilder):
    def __init__(self, sim_cls: type[magnetics.Simulation3DDifferential]):
        super().__init__(sim_cls)

    def build(self) -> magnetics.Simulation3DDifferential:
        return super().build()

    def receivers(self, receiver_points, components: str | Iterable[str] = 'tmi'):
        assert len(self._receiver_list) == 0, (
            '`Simulation3DDifferential` does not support'
            ' multiple groups of receiver points.'
        )
        super().receivers(receiver_points, components)


class Simulation3DDipolesBuilder(BasePFSimulationBuilder, MagneticSurveyBuilder):
    def __init__(self, sim_cls: type[metalpy_magnetics.Simulation3DDipoles]):
        super().__init__(sim_cls)
        self._survey_cls = metalpy_magnetics.VectorizedSurvey

        self.require_kwarg(mesh=False)  # 手动取消mesh参数

        self._source_list = []

    def build(self) -> metalpy_magnetics.Simulation3DDipoles:
        return super().build()

    def sources(self, pos, *other_pos):
        self._source_list.append(np.vstack([pos, *other_pos]))

    @SimulationBuilder._supplies('momentMap')
    def moment_map(self, chi_map: maps.IdentityMap | Dummy = undefined):
        return chi_map

    @SimulationBuilder._assembles('source_locations')
    def _source_locations(self):
        if self.has_mesh:
            self.sources(self.modelled_mesh.active_cell_centers)

        return np.vstack(self._source_list)
