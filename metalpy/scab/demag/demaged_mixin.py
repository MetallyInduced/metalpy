from __future__ import annotations

import contextlib

import numpy as np
from SimPEG import maps
from SimPEG.potential_fields.magnetics import Simulation3DIntegral
from discretize.utils import mkvc

from metalpy.mexin import Mixin
from metalpy.scab.demag.demagnetization import Demagnetization
from metalpy.scab.demag.factored_demagnetization import FactoredDemagnetization
from metalpy.scab.utils.misc import Field


class DemagedMapping(maps.IdentityMap):
    def __init__(self, simulation: Simulation3DIntegral, factor=None, kernel_dtype=None, **kwargs):
        assert simulation.model_type == 'scalar', '`Demaged` needs `scalar` model to work.'

        chiMap = simulation.chiMap

        super().__init__(nP=chiMap.shape[0])

        self.simulation = simulation
        self.factor = factor
        self.chiMap = chiMap

        if kernel_dtype is None:
            try:
                # simulation.sensitivity_dtype是 SimPEG 0.20 新增加的属性
                kernel_dtype = simulation.sensitivity_dtype
            except AttributeError:
                kernel_dtype = np.float64  # SimPEG < 0.20，默认使用float64

        if factor is not None:
            self.demag_solver = FactoredDemagnetization(n=factor)
        else:
            self.demag_solver = Demagnetization(
                mesh=self.simulation.mesh,
                active_ind=self.simulation.ind_active,
                kernel_dtype=kernel_dtype,
                **kwargs
            )

        self.cached_field = None
        self.cached_chi = None

    @property
    def shape(self):
        return self.nP * 3, self.nP

    def _transform(self, m):
        source_field = self.simulation.survey.source_field

        vec_model = self.get_cache(m, source_field)
        if vec_model is None:
            m = self.chiMap * m
            vec_model = self.demag_solver.dpred(m, source_field=source_field)
            self.cache(vec_model, field=source_field)

        return mkvc(vec_model)

    def inverse(self, D):
        raise NotImplemented

    def deriv(self, m, v=None):
        raise NotImplemented

    def cache(self, chi, field):
        self.cached_chi = np.copy(chi)
        self.cached_field = Field(field)

    def get_cache(self, model, field):
        if (
            isinstance(self.demag_solver, Demagnetization)
            and self.test_cached_field(field)
            and self.test_cached_model(model)
        ):
            return np.copy(self.cached_chi)
        else:
            return None

    def test_cached_field(self, field):
        return self.cached_field == Field(field)

    def test_cached_model(self, model):
        cached_model = self.demag_solver.solver.model
        return cached_model is not None and np.all(cached_model == model)


class DemagedMixin(Mixin):
    def __init__(
            self,
            this: Simulation3DIntegral,
            factor=None,
            **kwargs
    ):
        """初始化退磁计算上下文
        """
        super().__init__(this)
        this.chiMap = DemagedMapping(this, factor=factor, **kwargs)

    @Mixin.replaces(keep_orig='orig_fn')
    def linear_operator(self, _, orig_fn):
        with self._switch_model_type():
            return orig_fn()

    @contextlib.contextmanager
    def _switch_model_type(self, this: Simulation3DIntegral | None = None):
        """临时将模型类型切换为向量模型
        """
        this.model_type = 'vector'
        try:
            yield
        finally:
            this.model_type = 'scalar'
