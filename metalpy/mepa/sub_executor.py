import warnings
from typing import Sequence

from metalpy.utils.string import format_string_list
from .executor import Executor
from .worker import Worker


class SubExecutor(Executor):
    def __init__(self, parent: Executor, workers: Sequence[Worker]):
        """
        """
        super().__init__()
        self.workers = set(workers)
        self.parent = parent
        self.n_units = sum([w.weight for w in self.workers])

    def do_submit(self, func, *args, workers=None, **kwargs):
        if workers is not None:
            workers = set(workers)
            diff = workers - self.workers
            if len(diff) != 0:
                warnings.warn(
                    'Unexpected workers found. '
                    + format_string_list([f'`{w.name}`' for w in diff], multiline=True)
                )
        else:
            workers = self.workers
        return self.parent.do_submit(func, *args, workers=workers, **kwargs)

    def get_workers(self):
        return self.workers

    def get_n_units(self):
        return self.n_units

    def is_local(self):
        return self.parent.is_local()

    def _gather_single(self, future):
        return self.parent._gather_single(future)

    def scatter(self, data):
        return self.parent.scatter(data)

    def get_worker_context(self):
        return self.parent.get_worker_context()

    def receive_events(self, timeout):
        return self.parent.receive_events(timeout)
