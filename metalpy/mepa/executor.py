import contextlib
from abc import ABC, abstractmethod

from .task_allocator import TaskAllocator, SingleTaskAllocator
from .utils import structured_traverse


class Executor(ABC):
    """
    Note:
    1. 工作单元 n_units 以资源池中的基础算力作为单位。
        例：假设存在worker a和b，a的算力是b的两倍，则以b为基本单位，返回1+2=3
    """
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

    @abstractmethod
    def do_submit(self, func, *args, workers=None, **kwargs):
        pass

    def arrange_single(self, data):
        return SingleTaskAllocator(self.get_n_units(), data)

    def arrange(self, *data_list):
        return TaskAllocator(self.get_n_units(), *data_list)

    @abstractmethod
    def get_workers(self):
        pass

    @abstractmethod
    def get_n_units(self):
        """
        返回当前集群的基本工作单位数
        :return: 当前集群的基本工作单位数
        :rtype: int
        """
        pass

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
        3
        >>> executor.gather([f, [f], f])  # 支持列表
        [3, [3], 3]
        >>> executor.gather({'First': f, 'Second': [f, [f]]})  # 支持字典
        {'First': 3, 'Second': [3, [3]]}
        """
        with self.event_thread_if_necessary():
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
        with self.event_thread_if_necessary():
            return self._gather_single(future)

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
    def get_worker_context(self) -> 'WorkerContext':
        pass

    @abstractmethod
    def receive_events(self, timeout):
        pass

    def on(self, event, callback):
        self._event_handlers.setdefault(event, []).append(callback)

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
    def event_thread_if_necessary(self):
        try:
            self.start_event_monitoring(self.check_if_events_thread_are_required())
            yield
        finally:
            self.terminate_event_monitoring()


class WorkerContext:
    @abstractmethod
    def fire(self, event, msg):
        pass
