import queue
from concurrent.futures import Future
from functools import partial

from loky import get_reusable_executor

import psutil

from .executor import Executor, WorkerContext
from .utils import exception_caught, traverse_args
from .worker import Worker


class ProcessExecutor(Executor):
    def __init__(self, n_units=None):
        """构造基于ProcessPoolExecutor的执行器

        Parameters
        ----------
        n_units
            工作单位数，一般为物理核心数
        """
        super().__init__()
        if n_units is None:
            n_units = psutil.cpu_count(logical=False)

        self.pool = get_reusable_executor(max_workers=n_units, kill_workers=True)
        self.workers = [Worker(f'proc-{i}', 1) for i in range(n_units)]

        self._queue = None

    @property
    def queue(self):
        if self._queue is None:
            self._queue = self.pool._context.Manager().Queue()
        return self._queue

    @property
    def has_queue(self):
        return self._queue is not None

    def do_submit(self, func, *args, workers=None, **kwargs):
        """
        see also:
        ProcessPoolExecutor.submit
        """
        # 用于在func抛出异常时，打印异常栈信息然后传播
        # TODO: 让异常栈停留在异常抛出的位置，可能需要一些hack
        func = partial(exception_caught, func)

        # 自动将传入的future提取为值
        args, kwargs = traverse_args(args, kwargs,
                                     lambda x: x if not isinstance(x, Future) else self.gather([x])[0])
        return self.pool.submit(func, *args, **kwargs)

    def get_workers(self):
        return self.workers

    def is_local(self):
        return True

    def _gather_single(self, future):
        return future.result()

    def get_worker_context(self):
        return ProcessWorkerContext(self.queue)

    def receive_events(self, timeout):
        while True:
            try:
                event, msg = self.queue.get(True, timeout)
                yield event, msg
            except queue.Empty:
                yield None, None

    def check_if_events_thread_are_required(self):
        return (self.has_queue
                and super().check_if_events_thread_are_required())


class ProcessWorkerContext(WorkerContext):
    def __init__(self, queue):
        self.queue = queue

    def fire(self, event, msg):
        self.queue.put((event, msg))
