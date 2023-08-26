import uuid

import distributed
from distributed import Client

from .executor import Executor, WorkerContext
from .dask_helper import configure_dask_client
from .worker import Worker


class DaskExecutor(Executor):
    def __init__(self, scheduler_addr, extra_paths=None, excludes=None, upload_modules=True):
        """构造基于dask.distributed的执行器

        Parameters
        ----------
        scheduler_addr
            调度器地址，包括协议和端口号，例如 "tcp://xx.yy.zz:8786"
        extra_paths
            额外需要自动搜索并上传代码到集群的模块路径
        excludes
            需要排除不自动搜索并上传代码到集群的模块路径
        upload_modules
            指定是否自动搜索代码并上传到集群
        """
        super().__init__()
        self.client = Client(scheduler_addr)
        if upload_modules:
            configure_dask_client(self.client, extra_paths=extra_paths, excludes=excludes)

        worker_groups = {}
        for addr, worker in self.client.scheduler_info()['workers'].items():
            host = worker['host']
            if host not in worker_groups:
                worker_groups[host] = []
            # LocalCluster的名字有可能是数字，因此需要转换成字符串
            worker_groups[host].append(str(worker['name']))

        self.workers = []
        n_actual_working_units = 0
        load_weight = {}

        for host, workers in worker_groups.items():
            for worker in workers:
                worker_instance = Worker(worker, weight=1, verbose=False)
                worker_name = worker_instance.name

                if worker_name in load_weight:
                    weight = load_weight[worker_name]
                else:
                    weight = 1

                worker_instance.weight = weight
                self.workers.append(worker_instance)
                n_actual_working_units += weight

        self.client_id = str(uuid.uuid4())
        self._sub = None

    @property
    def sub(self):
        if self._sub is None:
            self._sub = distributed.Sub(self.client_id)

        return self._sub

    def do_submit(self, func, *args, workers=None, **kwargs):
        if isinstance(workers, Worker):
            workers = [workers]

        if workers is not None:
            targets = [worker.get_name() for worker in workers]
            kwargs['workers'] = targets
            kwargs.setdefault('allow_other_workers', True)

        return self.client.submit(func, *args, **kwargs)

    def get_workers(self):
        return self.workers

    def is_local(self):
        return False

    def _gather_single(self, future):
        return self.client.gather(future)

    def scatter(self, data):
        return self.client.scatter(data)

    def get_worker_context(self):
        return DaskWorkerContext(self.client_id)

    def receive_events(self, timeout):
        import asyncio

        while True:
            try:
                event, msg = self.sub.get(timeout)
                yield event, msg
            except asyncio.exceptions.TimeoutError:
                yield None, None


class DaskWorkerContext(WorkerContext):
    def __init__(self, client_id):
        self.client_id = client_id
        self._pub = None

    @property
    def pub(self):
        if self._pub is None:
            self._pub = distributed.Pub(self.client_id)

        return self._pub

    def fire(self, event, msg):
        self.pub.put((event, msg))
