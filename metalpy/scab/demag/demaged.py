from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Union

import numpy as np
from SimPEG import maps
from discretize.utils import mkvc

from metalpy.mexin import Mixin, mixin
from metalpy.mexin import Patch
from metalpy.scab import Tied
from metalpy.scab.demag import Demagnetization, FactoredDemagnetization
from metalpy.scab.distributed.policies import Distributable
from metalpy.scab.utils.misc import Field
from metalpy.utils.type import get_params_dict

if TYPE_CHECKING:
    from SimPEG.potential_fields.magnetics import Simulation3DIntegral


class DemagedMapping(maps.IdentityMap):
    def __init__(self, simulation: 'Simulation3DIntegral', factor=None, **kwargs):
        assert simulation.model_type == 'scalar', '`Demaged` needs `scalar` model to work.'

        chiMap = simulation.chiMap

        super().__init__(nP=chiMap.shape[0])

        self.simulation = simulation
        self.factor = factor
        self.chiMap = chiMap

        if factor is not None:
            self.demag_solver = FactoredDemagnetization(n=factor)
        else:
            self.demag_solver = Demagnetization(
                mesh=self.simulation.mesh,
                active_ind=self.simulation.ind_active,
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
            this: 'Simulation3DIntegral',
            factor=None,
            **kwargs
    ):
        """初始化退磁计算上下文
        """
        super().__init__(this)
        this.chiMap = DemagedMapping(this, factor=factor, **kwargs)

    @mixin.replaces(keep_orig='orig_fn')
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


class Demaged(Patch, Distributable):
    Priority = Tied.Priority + 1  # 保证在Tied后面注入mixin

    def __init__(
            self,
            factor=None,
            method=None,
            compressed_size: Union[int, float, None] = None,
            deterministic: Union[bool, str] = True,
            quantized: bool = False,
            progress: bool = False,
            **kwargs
    ):
        """为 `Simulation3DIntegral` 添加退磁效应计算

        Parameters
        ----------
        factor
            三轴退磁系数，如果为None，则采用数值方法求解
        method
            采用的数值求解算法，参考 `Demagnetization.methods`。默认为None，自动选择。
            可选选项：
                - Demagnetization.Compressed
                - Demagnetization.Seperated
                - Demagnetization.Integrated
        compressed_size
            压缩表尺寸，只针对`Demagnetization.Compressed`有效，用于指定压缩后的核矩阵哈希表最大大小，应尽可能小以满足内存要求。
            浮点数代表按完整核矩阵大小的比例设置，整数代表直接设置尺寸。
            如果大小不足或过大内部会给出提示信息
        deterministic
            采用确定性模式并行哈希，默认为True，牺牲一定计算效率和空间确保核矩阵正确。
            指定为CompressedForward.Optimal时直接采用unique函数计算理想压缩结果
        quantized
            采用量化模式压缩核矩阵，默认为False。
            True则启用压缩，使用更小位宽来存储核矩阵，减少一定的空间需求，略微牺牲计算效率。
            （可能会报要求尺寸为2的幂的性能警告）
        progress
            是否输出求解进度条，默认为False不输出
        """
        super().__init__()
        self.params = get_params_dict(
            factor=factor,
            method=method,
            compressed_size=compressed_size,
            deterministic=deterministic,
            quantized=quantized,
            progress=progress,
            **kwargs
        )

    def apply(self):
        from SimPEG.potential_fields.magnetics import Simulation3DIntegral
        self.add_mixin(Simulation3DIntegral, DemagedMixin, **self.params)

    def bind_context(self, ctx):
        super().bind_context(ctx)
