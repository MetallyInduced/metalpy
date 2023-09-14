from __future__ import annotations

import contextlib
import re
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Callable

from .task_allocator import BoundedTaskAllocator, ShuffleChoice
from .utils import structured_traverse
from .worker import Worker
from .worker_context import WorkerContext, BoundWorkerContext


class Executor(ABC):
    def __init__(self, event_interval=0.5):
        self.event_interval = event_interval
        self._running = False
        self._event_monitor_thread = None
        self._event_handlers: dict[str, list[callable]] = {}

        self._anonymous_event_counter = 0

    def submit(self, func, *args, worker=None, workers=None, **kwargs):
        workers = check_workers(worker, workers)
        return self.do_submit(func, workers=workers, *args, **kwargs)

    def map(self, func, *iterables, worker=None, workers=None, chunksize=None):
        workers = check_workers(worker, workers)

        futures = []
        n_workers = len(workers)

        for i, task in enumerate(zip(*iterables)):
            if workers is not None:
                w = [workers[(i // chunksize) % n_workers]]
            else:
                w = None
            futures.append(self.do_submit(func, *task, workers=w))

        return futures

    def extract_by_name(self, pat: str | re.Pattern | Callable):
        """筛选符合条件的worker构成子执行器

        Parameters
        ----------
        pat
            模式，支持通过字符串包含worker名/正则匹配worker名/函数谓词判定/重载`==`运算符等方式进行

        Returns
        -------
        sub_executor
            提取的worker构造的新的执行器
        """
        from .sub_executor import SubExecutor

        if isinstance(pat, str):
            workers = [w for w in self.get_workers() if pat in w.get_name()]
        elif isinstance(pat, re.Pattern):
            workers = [w for w in self.get_workers() if pat.match(w.get_name())]
        elif isinstance(pat, Callable):
            workers = [w for w in self.get_workers() if pat(w)]
        else:
            workers = [w for w in self.get_workers() if pat == w]

        return SubExecutor(parent=self, workers=workers)

    @abstractmethod
    def do_submit(self, func, *args, workers=None, **kwargs):
        pass

    def arrange(self, data, shuffle: ShuffleChoice = False) -> BoundedTaskAllocator:
        """将给定序列数据按当前worker的权重分割

        Parameters
        ----------
        data
            序列数据
        shuffle
            指示是否打乱数据

        Returns
        -------
        allocator
            数据分配器，遍历即可得到按权重分配的数据
        """
        return BoundedTaskAllocator.arrange(
            data,
            weights=self.get_workers(),
            shuffle=shuffle
        )

    def arrange_many(self, *data_list, shuffle: ShuffleChoice = False) -> BoundedTaskAllocator:
        """将给定多个序列数据按当前worker的权重分割，并类似zip(*)进行组合

        Parameters
        ----------
        data_list
            多个序列数据
        shuffle
            指示是否打乱数据

        Returns
        -------
        allocator
            数据组分配器，遍历即可得到按权重分配的数据组

        Examples
        --------
        >>> import numpy as np
        >>> from metalpy.mepa import ThreadExecutor
        >>> executor = ThreadExecutor(3)
        >>> tasks = executor.arrange_many(np.arange(7), np.arange(7, 14))
        >>> print(list(tasks))
        <<< [(array([0, 1, 2]), array([7, 8, 9])),
        <<<  (array([3, 4, 5]), array([10, 11, 12])),
        <<<  (array([6]), array([13]))]
        """
        return BoundedTaskAllocator.arrange_many(
            *data_list,
            weights=self.get_workers(),
            shuffle=shuffle
        )

    @abstractmethod
    def get_workers(self) -> Collection[Worker]:
        pass

    def get_n_workers(self):
        """返回当前集群的总worker数
        """
        return len(self.get_workers())

    def get_total_weights(self):
        """返回当前集群所有worker的权重和
        """
        return sum([w.get_weight() for w in self.get_workers()])

    @abstractmethod
    def is_local(self):
        return True

    def gather(self, futures):
        """收集futures的结果

        Parameters
        ----------
        futures
            待收集的futures，可以为单个future也可以是future的Iterable或dict

        Returns
        -------
        ret
            以原本结构构造的future结果

        Examples
        --------
        >>> from operator import add
        >>> from metalpy.mepa import LinearExecutor
        >>> executor = LinearExecutor()
        >>> f = executor.submit(add, 1, 2)
        >>> executor.gather(f)
        <<< 3
        >>> executor.gather([f, [f], f])  # 支持列表
        <<< [3, [3], 3]
        >>> executor.gather({'First': f, 'Second': [f, [f]]})  # 支持字典
        <<< {'First': 3, 'Second': [3, [3]]}
        """
        with self.monitor_events():
            return self._gather(futures)

    def gather_single(self, future):
        """收集单个的结果

        Parameters
        ----------
        future
            待收集的future

        Returns
        -------
        ret
            future的结果
        """
        with self.monitor_events():
            return self._gather_single(future)

    def submit_and_gather(self, func, *args, worker=None, workers=None, **kwargs):
        fut = self.submit(func, *args, worker=worker, workers=workers, **kwargs)
        return self.gather(fut)

    def _gather(self, futures):
        """收集futures的结果的实现逻辑

        Parameters
        ----------
        futures
            待收集的futures，可以为单个future也可以是future的Iterable或dict

        Returns
        -------
        ret
            以原本结构构造的future结果

        Notes
        -----
            子类需从_gather和_gather_single二选一实现
            子类实现_gather时建议实现structured_traverse逻辑
        """
        return structured_traverse(futures, lambda f: self._gather_single(f))

    def _gather_single(self, future):
        """收集单个的结果的实现逻辑

        Parameters
        ----------
        future
            待收集的future

        Returns
        -------
        ret
            future的结果

        Notes
        -----
            子类需从_gather和_gather_single二选一实现
        """
        return self._gather(future)

    def scatter(self, data):
        """
        用于预先分发大规模数据（一般指内存占用大于1MB）

        Parameters
        ----------
        data
            大规模数据

        See Also
        --------
        distributed.Client.scatter
        """
        return data

    @abstractmethod
    def get_worker_context(self) -> WorkerContext:
        """获取当前Executor绑定的WorkerContext实例

        Returns
        -------
        context
            返回当前Executor绑定的WorkerContext实例，如果没有，则创建

        Notes
        -----
        一般情况下，所有事件共享一个WorkerContext实例，发布消息时由Executor完成消息分发
        """
        pass

    @abstractmethod
    def receive_events(self, timeout):
        pass

    def on(self, event, callback) -> BoundWorkerContext:
        """绑定子worker发出的消息事件回调。

        Executor会在单独的监听线程中获取消息事件并调用回调函数。

        Parameters
        ----------
        event
            监听的事件标记或事件回调函数
        callback
            事件回调函数

        Returns
        -------
        context
            返回绑定到该事件的WorkerContext，可以直接通过fire调用到该回调函数，不需要指定事件
        """
        self._event_handlers.setdefault(event, []).append(callback)
        return self.get_worker_context().bind(event)

    def register(self, callback) -> BoundWorkerContext:
        """绑定一个匿名消息的事件回调。

        Parameters
        ----------
        callback
            事件回调函数

        Returns
        -------
        context
            返回绑定到该事件的WorkerContext，可以直接通过fire调用到该回调函数，不需要指定事件

        See Also
        --------
        Executor.on : 会自动生成一个专有名字然后通过on绑定
        """
        event = _Anonymous(self._anonymous_event_counter)
        self._anonymous_event_counter += 1

        return self.on(event, callback)

    def dispatch_event(self, event, args, kwargs):
        listeners = self._event_handlers.get(event, None)
        if listeners is not None:
            for listener in listeners:
                listener(*args, **kwargs)

    def monitor(self):
        for event, msg in self.receive_events(self.event_interval):
            if event is None:
                pass
            else:
                args, kwargs = msg
                self.dispatch_event(event, args, kwargs)

            if not self._running:
                return

    def start_event_monitoring(self, start_thread):
        self._running = True
        if start_thread:
            from threading import Thread
            self._event_monitor_thread = Thread(target=self.monitor)
            self._event_monitor_thread.start()

    def terminate_event_monitoring(self):
        self._running = False

    def check_if_events_thread_are_required(self):
        return len(self._event_handlers) > 0

    @contextlib.contextmanager
    def monitor_events(self):
        """启动worker消息事件的监听线程

        Notes
        -----
        `gather`和`gather_single`会自动调用该函数启动监听线程、
        """
        if not self._running:
            try:
                self.start_event_monitoring(self.check_if_events_thread_are_required())
                yield
            finally:
                self.terminate_event_monitoring()
        else:
            try:
                yield
            finally:
                pass

    def shutdown(self, wait=True):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)


class _Anonymous:
    __slots__ = ['i']

    def __init__(self, i):
        self.i = i

    def __hash__(self):
        return hash(f'__A{self.i}')

    def __eq__(self, other):
        return isinstance(other, _Anonymous) and self.i == other.i


def check_workers(worker, workers):
    if workers is None:
        if worker is None:
            workers = None
        else:
            workers = [worker]
    else:
        workers = list(workers)

    return workers
