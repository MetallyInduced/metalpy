from __future__ import annotations

import os
import sys

import numpy as np
from SimPEG.simulation import BaseSimulation
from discretize import TreeMesh

from metalpy.mexin import Patch
from metalpy.mexin.injectors import replaces
from metalpy.mexin.mixins import DispatcherMixin
from metalpy.scab.distributed.policies import Distributable
from metalpy.utils.type import get_params_dict
from .taichi_kernel_base import Profiler


class _Tied(DispatcherMixin, allow_match_parent=True, warns_when_not_matched=True):
    def __init__(self, this, max_cpu_threads=None, profile: Profiler | bool = False, **kwargs):
        from metalpy.utils.taichi import ti_init

        params = {}

        profile = Profiler.of(profile)
        self.profile = profile
        if profile.needs_kernel_profiler:
            params['kernel_profiler'] = True

        if max_cpu_threads is not None:
            if max_cpu_threads < 0:
                # 需要注意taichi目前使用的为虚拟核心数 (taichi/program/compile_config.cpp)
                # cpu_max_num_threads = std::thread::hardware_concurrency();
                # 因此这里使用python自带的cpu_count()，而不是psutil的物理核心数，和taichi保证一致
                # 考虑到超线程的存在，-1可能仍然会导致计算机满载
                max_cpu_threads = os.cpu_count() + max_cpu_threads
            params['cpu_max_num_threads'] = max_cpu_threads

        ti_init(**params, **kwargs)
        super().__init__(this, profile=self.profile)


class Tied(Patch, Distributable):
    def __init__(self, arch=None, max_cpu_threads=None, profile: Profiler | bool = False, **kwargs):
        """将正演算子替换为基于Taichi的并行正演方法

        Parameters
        ----------
        arch
            taichi的运行设备，默认为 `cpu` ，
            可以通过字符串指定（例如 `'cpu'` 或 `'gpu'` ），
            也可以通过 `ti.cpu` 或 `ti.gpu` 指定
        max_cpu_threads
            cpu arch下的最大线程数，taichi默认为cpu核心数
        profile
            指示是否启用profiler，详见 :class:`Profiler`
        """
        super().__init__()
        self.params = get_params_dict(
            arch=arch,
            max_cpu_threads=max_cpu_threads,
            profile=profile,
            **kwargs
        )

    def apply(self):
        self.add_mixin(BaseSimulation, _Tied, **self.params)

        if 'SimPEG.potential_fields.base' in sys.modules:
            # BasePFSimulation构造函数中会预先计算node unique，而目前的Taichi版本正演不需要相关信息，因此禁用以提高性能
            from SimPEG.potential_fields.base import BasePFSimulation
            self.add_injector(
                replaces(BasePFSimulation.__init__, keep_orig='orig'),
                Tied.disable_pf_sim_node_preprocessing
            )

    @staticmethod
    def disable_pf_sim_node_preprocessing(this, mesh, ind_active=None, *args, orig, **kwargs):
        # 使用空网格替换原网格，快速跳过初始化步骤
        _mesh = TreeMesh([[1, 0]] * 3)
        _ind_active = np.ones(_mesh.n_cells, dtype=bool)

        orig(this, _mesh, _ind_active, *args, **kwargs)

        # 还原为原网格和有效索引
        this.mesh = mesh
        this._ind_active = ind_active
        this.nC = int(ind_active.sum())


@_Tied.implements('SimPEG.potential_fields.magnetics.simulation:Simulation3DIntegral')
def _():
    from .potential_fields.magnetics.simulation import TiedSimulation3DIntegralMixin
    return TiedSimulation3DIntegralMixin
