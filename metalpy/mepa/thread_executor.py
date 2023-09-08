import queue
from concurrent.futures import Future, ThreadPoolExecutor

import psutil

from .executor import Executor, WorkerContext
from .utils import traverse_args
from .worker import Worker


class ThreadExecutor(Executor):
    def __init__(self, n_units=None):
        """构造基于ThreadPoolExecutor的执行器

        Parameters
        ----------
        n_units
            工作单位数，一般为物理核心数
        """
        super().__init__()
        if n_units is None:
            n_units = psutil.cpu_count(logical=False)

        self.pool = ThreadPoolExecutor(max_workers=n_units)
        self.workers = [Worker(f'thread-{i}', 1) for i in range(n_units)]

        self._queue = None

    @property
    def queue(self):
        if self._queue is None:
            self._queue = queue.Queue()
        return self._queue

    @property
    def has_queue(self):
        return self._queue is not None

    def do_submit(self, func, *args, workers=None, **kwargs):
        """
        See Also
        --------
        ThreadPoolExecutor.submit : 线程池提交任务
        """
        # 自动将传入的future提取为值
        args, kwargs = traverse_args(
            args, kwargs,
            lambda x: x if not isinstance(x, Future) else self.gather(x)
        )
        return self.pool.submit(func, *args, **kwargs)

    def get_workers(self):
        return self.workers

    def is_local(self):
        return True

    def _gather_single(self, future):
        return future.result()

    def get_worker_context(self):
        return ThreadWorkerContext(self.queue)

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown(wait=True)


class ThreadWorkerContext(WorkerContext):
    def __init__(self, queue):
        self.queue = queue

    def fire(self, event, msg):
        self.queue.put((event, msg))
