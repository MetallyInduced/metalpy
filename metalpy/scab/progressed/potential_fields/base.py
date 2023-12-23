import numpy as np
from SimPEG.potential_fields.base import BasePFSimulation

from metalpy.mexin import Mixin
from metalpy.scab.progressed import Progress


class ProgressedBasePFSimulationMixin(Mixin):
    def __init__(self, this, *_, **__):
        super().__init__(this)
        self.progress = this.mixins.get(Progress)

    @Mixin.replaces(keep_orig='orig')
    def linear_operator(self, this: BasePFSimulation, orig):
        assert this.n_processes == 1, (
            f'`Progressed` patch does not support multiprocessing mode'
            f' (n_processes = {this.n_processes} != 1).'
            f' Consider using `Distributed(...)` patch for parallel simulation instead.'
        )

        total = this.survey.nD
        if this.store_sensitivities != 'forward_only':
            total *= this.nC
            if getattr(this, 'model_type', 'scalar') == 'vector':
                total *= 3

        self.progress.reset(total=total)
        ret = orig()
        self.progress.close()

        return ret

    @Mixin.after(keep_retval='retval')
    def evaluate_integral(self, _, *__, retval, **___):
        self.progress.update(np.prod(retval.shape))
        return retval
