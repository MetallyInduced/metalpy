import abc
import queue
from concurrent.futures import Future
from concurrent import futures
from functools import partial

from .executor import Executor
from .queue_like_worker_context import QueueLikeWorkerContext
from .utils import exception_caught, traverse_args


class PoolExecutor(Executor):
    def __init__(self, pool_executor: futures.Executor, workers):
        """构造基于Python标准Executor的执行器

        Parameters
        ----------
        pool_executor
            符合Python约定的工作单元池执行器
        """
        super().__init__()
        self.pool = pool_executor
        self.workers = workers

        self._queue = None

    @abc.abstractmethod
    def _get_queue(self):
        pass

    @property
    def queue(self):
        if self._queue is None:
            self._queue = self._get_queue()
        return self._queue

    @property
    def has_queue(self):
        return self._queue is not None

    def map(self, func, *iterables, worker=None, workers=None, chunksize=None):
        with self.monitor_events():
            return list(self.pool.map(func, *iterables, chunksize=chunksize))

    def do_submit(self, func, *args, workers=None, **kwargs):
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
        return QueueLikeWorkerContext(self.queue)

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

    def shutdown(self, wait=True):
        self.pool.shutdown(wait=wait)
