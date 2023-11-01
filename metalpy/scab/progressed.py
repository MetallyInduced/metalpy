from __future__ import annotations

from typing import Union

import tqdm
from SimPEG.simulation import BaseSimulation

from metalpy.mepa import Executor, Worker, ParallelProgress
from metalpy.mexin import Mixin
from metalpy.mexin.patch import Patch
from .distributed.policies import Distributable

ProgressContext = Union[tqdm.tqdm, ParallelProgress, None]


class Progress(Mixin):
    def __init__(self, this: BaseSimulation, patch: Progressed):
        super().__init__(this)
        self.patch = patch

    def get_progress_ctx(self, _=None):
        return self.patch.get_context()

    def reset(self, _, total):
        self.patch.reset(total)

    def update(self, _, n):
        self.get_progress_ctx().update(n)

    def close(self, _):
        self.get_progress_ctx().close()


class Progressed(Patch, Distributable):
    BaseProgressStyles = {
        'unit_scale': True,
        'smoothing': 0.1,
    }

    def __init__(self, _progress_ctx: ProgressContext = None, **tqdm_kw):
        """进度条插件，为正演添加进度条机制（目前）

        Parameters
        ----------
        _progress_ctx
            进度上下文，用户不应手动指定
        tqdm_kw
            用于 `tqdm` 的进度条参数
        """
        super().__init__()
        self.progress_ctx: ProgressContext = _progress_ctx
        self.tqdm_kw = {**Progressed.BaseProgressStyles, **tqdm_kw}

    def apply(self):
        if len(self.context.get_patches()) == 1:
            raise RuntimeError("Progressed patch cannot be used alone due to SimPEG's design")
        self.add_mixin(BaseSimulation, Progress, patch=self)

    def get_context(self, executor: Executor | None = None, allow_restart=False):
        if self.progress_ctx is None or (self.progress_ctx.disable and allow_restart):
            if executor is not None:
                self.progress_ctx = executor.progress(**self.tqdm_kw)
            else:
                self.progress_ctx = tqdm.tqdm(**self.tqdm_kw)
        return self.progress_ctx

    def reset(self, total):
        # 允许重启进度条以支持重用 Progressed 实例
        self.get_context(allow_restart=True).reset(total)

    def distribute_to(self, executor: Executor, worker: Worker):
        # 允许重启进度条以支持重用 Progressed 实例
        return Progressed(_progress_ctx=self.get_context(executor, allow_restart=True))
