import numpy as np

from metalpy.utils.batch import Batch
from .worker import Worker


class SingleTaskAllocator:
    def __init__(self, data, total=None):
        self.data = data
        if total is None:
            total = len(data)
        self.total = total
        self.base_index = 0

    def slice(self, n_slices=1):
        end = self.base_index + int(np.ceil(len(self.data) * n_slices / self.total))
        ret = self.data[self.base_index:end]
        self.base_index = end
        return ret

    def assign(self, worker: Worker):
        return self.slice(worker.get_weight())

    @property
    def finished(self):
        return self.base_index >= len(self.data)


class TaskAllocator:
    def __init__(self, *data_list, total=None):
        allocators = [SingleTaskAllocator(data, total=total) for data in data_list]
        self.allocators = Batch.of(allocators)

    def slice(self, n_slices=1):
        return list(self.allocators.slice(n_slices))

    def assign(self, worker: Worker):
        return list(self.allocators.assign(worker))

    @property
    def finished(self):
        return any(self.allocators.finished)
