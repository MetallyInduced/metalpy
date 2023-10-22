from __future__ import annotations

import contextlib
import functools
from threading import Lock
from typing import TYPE_CHECKING, Optional, Sequence, TypeVar, Generator, overload

import tqdm


if TYPE_CHECKING:
    from metalpy.mepa import Executor, WorkerContext


_ElemT = TypeVar('_ElemT')


def _locally_locked(func: _ElemT) -> _ElemT:
    @functools.wraps(func)
    def wrapper(self: ParallelProgress, *args, **kwargs):
        if self._local_lock is None:
            return func(self, *args, **kwargs)
        else:
            with self._local_lock:
                return func(self, *args, **kwargs)

    return wrapper


class ParallelProgress:
    def __init__(self, executor: Optional['Executor'], *, total=0, **kwargs):
        """适用于 `mepa.Executor` 的并行进度条，
        支持直接传递给worker，以类似 `tqdm` 的形式并发更新进度

        如果初始化时指定了 `total` ，则创建者也有义务执行 `close` 进行关闭。

        Parameters
        ----------
        executor
            执行器
        total
            任务总量，也可以在各个worker中用 `register` 登记工作量
        kwargs
            其它要传给 `tqdm` 的关键字参数

        Notes
        -----
        Executor本地的所有事件响应是在单线程内进行的，
        因此如果对象通过WorkerContext通信更新进度的话，是不需要加锁的，
        需要创建时手动指定 `enable_local_lock(False)`

        但是例如ThreadExecutor这种本地的执行器，
        可以绕过WorkerContext通信更新进度，
        此时则需要对操作进行加锁，防止进度条数据错误

        Examples
        --------
        基本函数包含 `register` 、 `register` 和 `close` ，分别负责注册进度，更新进度和汇报进度完成。
        其中 `register` 和 `close` 应保证成对出现：

        >>> with ProcessExecutor(n_tasks) as executor:
        >>>     progress = executor.progress()
        >>>     def task():
        >>>         progress.register(10)  # 告知主机工作量
        >>>         for i in range(10):
        >>>             sleep(0.5)
        >>>             progress.update(1)  # 告知主机进度更新
        >>>         progress.close()  # 告知主机工作已完成（每个register应当与close配对）
        >>>     tasks = [executor.submit(task) for _ in range(n_tasks)]
        >>>     executor.gather(tasks)

        可以通过 `progress` 提供的上下文管理器来避免手动 `close` ：

        >>> with ProcessExecutor(4) as executor:
        >>>     progress = executor.progress()
        >>>     def task():
        >>>         with progress(10):
        >>>             for i in range(10):
        >>>                 sleep(0.5)
        >>>                 progress.update(1)  # 告知主机进度更新
        >>>     tasks = [executor.submit(task) for _ in range(4)]
        >>>     executor.gather(tasks)

        `progress` 也提供了类似 `tqdm` 的便捷函数 `range` 和 `iter` （ `trange` 和 `tqdm` ）：

        >>> with ProcessExecutor(4) as executor:
        >>>     progress = executor.progress()
        >>>     def task():
        >>>         for _ in progress.range(10):
        >>>             sleep(0.5)
        >>>     tasks = [executor.submit(task) for _ in range(4)]
        >>>     executor.gather(tasks)
        """
        self.executor = executor
        self.progress = None

        self.progress_configs = {
            'total': total,
            **kwargs
        }
        self.progress: tqdm.tqdm | None = None

        self.context: WorkerContext | None = None
        self._register_key = None
        self._update_key = None
        self._close_key = None

        if executor is not None:
            self.context = executor.get_worker_context()
            self._register_key = executor.register(self.register).event
            self._update_key = executor.register(self.update).event
            self._close_key = executor.register(self.close).event
            self.progress = tqdm.tqdm(**self.progress_configs)

        self.n_distributes = 0
        if total > 0:
            # 创建者制定了总量，也视为一次分发，需要由创建者来 `close`
            self.n_distributes += 1

        self._local_lock = Lock()

    @contextlib.contextmanager
    def __call__(self, total):
        self.register(total)
        try:
            yield
        finally:
            self.close()

    def iters(self, iterable: Sequence[_ElemT]) -> Generator[_ElemT, None, None]:
        total = len(iterable)
        with self(total):
            for elem in iterable:
                yield elem
                self.update(1)

    @overload
    def range(self, end): ...

    @overload
    def range(self, start, end): ...

    @overload
    def range(self, start, end, step): ...

    def range(self, *args):
        yield from self.iters(range(*args))

    @_locally_locked
    def register(self, n):
        if self.executor is not None:
            self.n_distributes += 1
            self.progress.total += n
            self.progress.refresh()
        else:
            self.context.fire(self._register_key, n)

    @_locally_locked
    def update(self, n):
        if self.executor is not None:
            self.progress.update(n)
        else:
            self.context.fire(self._update_key, n)

    @_locally_locked
    def close(self):
        if self.executor is not None:
            if self.n_distributes > 0:
                self.n_distributes -= 1
                if self.n_distributes <= 0 and self.progress is not None:
                    self.dispose()
        else:
            self.context.fire(self._close_key)

    tqdm = iters
    trange = range
    reset = register

    def enable_local_lock(self, enable):
        """指定是否对进度条操作操作启用锁

        Parameters
        ----------
        enable
            指示是否启用
        """
        if enable:
            if self._local_lock is None:
                self._local_lock = Lock()
        else:
            self._local_lock = None

    def dispose(self):
        if self.executor is not None and self.context is not None:
            self.progress.close()
            self._register_key = self.executor.off(self._register_key)
            self._update_key = self.executor.off(self._update_key)
            self._close_key = self.executor.off(self._close_key)
            self.context = None

    def distribute(self):
        self.enable_local_lock(False)

        return {
            'executor': None,
            'context': self.context,
            '_register_key': self._register_key,
            '_update_key': self._update_key,
            '_close_key': self._close_key,
            '_local_lock': None
        }

    def __getstate__(self):
        return self.distribute()
