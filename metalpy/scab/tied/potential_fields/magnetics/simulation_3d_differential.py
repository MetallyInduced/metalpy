from __future__ import annotations

from SimPEG.potential_fields.magnetics import Simulation3DDifferential

from metalpy.scab.solvers import TiedSparseSolver
from ...taichi_kernel_base import TiedMixin, Profiler


class TiedSimulation3DDifferentialMixin(TiedMixin):
    def __init__(self, this: Simulation3DDifferential, profile: Profiler | bool):
        super().__init__(this, profile)

        this.solver_opts = {
            'fallback': this.solver,
            'fallback_opts': this.solver_opts,
            'rtol': 1e-5
        }
        this.solver = TiedSparseSolver
