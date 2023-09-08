from __future__ import annotations

import contextlib
import re
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Callable

from . import Worker
from .task_allocator import TaskAllocator, SingleTaskAllocator
from .utils import structured_traverse


class Executor(ABC):
    def __init__(self, event_interval=0.5):
        self.event_interval = event_interval
        self._running = False
        self._event_monitor_thread = None
        self._event_handlers: dict[str, list[callable]] = {}

    def submit(self, func, *args,
               worker=None, workers=None, **kwargs):
        if workers is None and worker is not None:
            workers = [worker]
        return self.do_submit(func, workers=workers, *args, **kwargs)

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

    def arrange(self, data):
        return SingleTaskAllocator(data, total=self.get_total_weights())

    def arrange_many(self, *data_list):
        return TaskAllocator(*data_list, total=self.get_total_weights())

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
        >>> from metalpy.mepa.linear_executor import LinearExecutor
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
            监听的事件标记
        callback
            事件回调函数

        Returns
        -------
        context
            返回绑定到该事件的WorkerContext
        """
        self._event_handlers.setdefault(event, []).append(callback)
        return self.get_worker_context().bind(event)

    def dispatch_event(self, event, msg):
        listeners = self._event_handlers.get(event, None)
        for listener in listeners:
            listener(msg)

    def monitor(self):
        for event, msg in self.receive_events(self.event_interval):
            if event is None:
                pass
            else:
                self.dispatch_event(event, msg)

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
        try:
            self.start_event_monitoring(self.check_if_events_thread_are_required())
            yield
        finally:
            self.terminate_event_monitoring()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class WorkerContext(ABC):
    @abstractmethod
    def fire(self, event, msg):
        pass

    def bind(self, event):
        return BoundWorkerContext(self, event)


class BoundWorkerContext:
    def __init__(self, context: WorkerContext, event):
        self.context = context
        self.event = event

    def fire(self, msg):
        self.context.fire(self.event, msg)
