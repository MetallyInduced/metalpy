import queue
from concurrent.futures import ThreadPoolExecutor

import psutil

from .pool_executor import PoolExecutor
from .worker import Worker


class ThreadExecutor(PoolExecutor):
    def __init__(self, n_units=None):
        """构造基于ThreadPoolExecutor的执行器

        Parameters
        ----------
        n_units
            工作单位数，一般为物理核心数
        """
        if n_units is None:
            n_units = psutil.cpu_count(logical=False)

        pool = ThreadPoolExecutor(max_workers=n_units)
        workers = [Worker(f'thread-{i}', 1) for i in range(n_units)]

        super().__init__(pool_executor=pool, workers=workers)

    def _get_queue(self):
        return queue.Queue()
