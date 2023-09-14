from __future__ import annotations

import tqdm
from SimPEG.simulation import BaseSimulation

from metalpy.mepa import Executor, Worker
from metalpy.mexin import Mixin
from metalpy.mexin.patch import Patch
from .distributed.policies import Distributable
from ..mepa.executor import WorkerContext


class Progress(Mixin):
    def __init__(self, this: BaseSimulation, ctx: WorkerContext = None):
        super().__init__(this)

        self.progress_ctx = ctx
        self.progressbar: tqdm.tqdm | None = None

    @property
    def progressbar(self) -> tqdm.tqdm | None:
        return self._progressbar

    @progressbar.setter
    def progressbar(self, value: tqdm.tqdm | None):
        self._progressbar = value

    def get_progressbar(self, this):
        return self.progressbar

    def reset(self, this, total):
        if self.progress_ctx is not None:
            self.progress_ctx.fire(Progressed.REGISTER, total)
        else:
            self.progressbar = tqdm.tqdm(total=total, **Progressed.PROGRESS_STYLES)

    def update(self, this, count):
        if self.progress_ctx is not None:
            self.progress_ctx.fire(Progressed.UPDATE, count)
        elif self.progressbar is not None:
            self.progressbar.update(count)

    def close(self, this):
        if self.progress_ctx is not None:
            self.progress_ctx.fire(Progressed.FINISHED, 1)
        elif self.progressbar is not None:
            self.progressbar.close()


class Progressed(Patch, Distributable):
    REGISTER = 'Progressed-register'
    UPDATE = 'Progressed-update'
    FINISHED = 'Progressed-finished'
    PROGRESS_STYLES = {
        'unit_scale': True
    }

    def __init__(self, ctx=None):
        super().__init__()
        self.ctx = ctx
        self.progress: tqdm.tqdm | None = None
        self.n_distributes = 0

    def apply(self):
        if len(self.context.get_patches()) == 1:
            raise RuntimeError("Progressed patch cannot be used alone due to SimPEG's design")

        self.add_mixin(BaseSimulation, Progress, ctx=self.ctx)

    def get_context(self, executor: Executor):
        if self.ctx is None:
            self.progress = tqdm.tqdm(total=0, **Progressed.PROGRESS_STYLES)
            executor.on(Progressed.REGISTER, self.add_to_total)
            executor.on(Progressed.UPDATE, lambda x: self.progress.update(x))
            executor.on(Progressed.FINISHED, self.indicate_finished)
            self.ctx = executor.get_worker_context()

        self.n_distributes += 1
        return self.ctx

    def add_to_total(self, i):
        self.progress.total += i
        self.progress.refresh()

    def indicate_finished(self, i):
        self.n_distributes -= i
        if self.n_distributes <= 0 and self.progress is not None:
            self.progress.close()

    def distribute_to(self, executor: Executor, worker: Worker):
        return Progressed(ctx=self.get_context(executor))
