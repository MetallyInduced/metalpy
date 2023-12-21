from metalpy.mexin import Mixin, mixin
from metalpy.scab.progressed import Progress


class ProgressedBasePFSimulationMixin(Mixin):
    def __init__(self, this, *_, **__):
        super().__init__(this)
        self.progress = this.mixins.get(Progress)

    @mixin.replaces(keep_orig='orig')
    def linear_operator(self, this, orig):
        assert this.n_processes == 1, (
            f'`Progressed` patch does not support multiprocessing mode'
            f' (n_processes = {this.n_processes} != 1).'
            f' Consider using `Distributed(...)` patch for parallel simulation instead.'
        )

        self.progress.reset(total=this.survey.nD)
        ret = orig()
        self.progress.close()

        return ret

    @mixin.after(keep_retval='retval')
    def evaluate_integral(self, _, *__, retval, **___):
        self.progress.update(retval.shape[0])
        return retval
