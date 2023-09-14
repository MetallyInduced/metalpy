import psutil
from loky import get_reusable_executor

from .pool_executor import PoolExecutor
from .worker import Worker


class ProcessExecutor(PoolExecutor):
    def __init__(self, n_units=None):
        """构造基于loky进程池的执行器

        Parameters
        ----------
        n_units
            工作单位数，一般为物理核心数

        See Also
        --------
        :class:`ProcessPoolExecutor` : 原生进程池
        """
        if n_units is None:
            n_units = psutil.cpu_count(logical=False)

        pool = get_reusable_executor(max_workers=n_units, kill_workers=True)
        workers = [Worker(f'proc-{i}', 1) for i in range(n_units)]

        super().__init__(pool_executor=pool, workers=workers)

    def _get_queue(self):
        return self.pool._context.Manager().Queue()
