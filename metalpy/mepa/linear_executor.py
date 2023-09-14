import psutil

from .executor import Executor, WorkerContext
from .lazy_evaluator import LazyEvaluator
from .utils import traverse_args
from .worker import Worker


class LinearExecutor(Executor):
    def __init__(self, n_units=None):
        """
        在当前进程上构造基于单线程执行器，常用于调试
        :param n_units: 工作单位数
        """
        super().__init__()
        if n_units is None:
            n_units = psutil.cpu_count(logical=False)

        self.workers = [Worker(f'local-{i}', 1) for i in range(n_units)]

    def do_submit(self, func, *args, workers=None, **kwargs):
        # 自动提取所有future
        args, kwargs = traverse_args(args, kwargs,
                                     lambda x: x if not isinstance(x, LazyEvaluator) else self.gather([x])[0])
        return LazyEvaluator(func, *args, **kwargs)

    def get_workers(self):
        return self.workers

    def is_local(self):
        return True

    def _gather_single(self, future):
        return future.result()

    def get_worker_context(self):
        return LinearWorkerContext(self)

    def receive_events(self, timeout):
        pass

    def check_if_events_thread_are_required(self):
        return False


class LinearWorkerContext(WorkerContext):
    def __init__(self, linear_client: LinearExecutor):
        self.linear_client = linear_client

    def fire(self, event, *args, **kwargs):
        self.linear_client.dispatch_event(event, args, kwargs)
