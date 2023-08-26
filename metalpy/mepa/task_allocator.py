import numpy as np


class SingleTaskAllocator:
    def __init__(self, total, data):
        self.data = data
        self.total = total
        self.base_index = 0

    def slice(self, n_slices=1):
        end = self.base_index + int(np.ceil(len(self.data) * n_slices / self.total))
        ret = self.data[self.base_index:end]
        self.base_index = end
        return ret

    def assign(self, worker):
        return self.slice(worker.get_weight())


class TaskAllocator:
    def __init__(self, total, *data_list):
        self.allocators = []
        for data in data_list:
            self.allocators.append(SingleTaskAllocator(total, data))

    def assign(self, worker):
        ret = []
        for allocator in self.allocators:
            ret.append(allocator.assign(worker))
        return ret
