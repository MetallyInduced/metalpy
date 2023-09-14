from __future__ import annotations

import abc
import warnings
from typing import Iterable, TYPE_CHECKING, Union

import numpy as np

from metalpy.utils.batch import Batch
from metalpy.utils.rand import check_random_state
from .worker import Worker

if TYPE_CHECKING:
    from .executor import Executor

WeightsType = Union[Iterable[Union[Worker, float]], 'Executor']
ShuffleChoice = Union[bool, int, np.random.RandomState, np.ndarray]


class TaskAllocator(abc.ABC):
    @abc.abstractmethod
    def slice(self, n_slices=1):
        pass

    @abc.abstractmethod
    def assign(self, worker_or_weight: Worker | float):
        pass

    @abc.abstractmethod
    def iter(self, var_inp: WeightsType):
        pass

    @property
    @abc.abstractmethod
    def finished(self):
        pass

    @abc.abstractmethod
    def inverse_shuffle(self, *data):
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass


class SingleTaskAllocator(TaskAllocator):
    def __init__(self, data, *, total=None, shuffle: ShuffleChoice = False):
        self.data = data
        if total is None:
            total = len(data)
        self.total = total
        self.base_index = 0

        if shuffle:
            if isinstance(shuffle, np.ndarray):
                assert len(shuffle) == len(data), \
                    '`shuffle` should be random mask when provided with ndarray.'
                self.shuffle_mask = shuffle
            else:
                self.shuffle_mask = np.arange(len(data))
                if shuffle is True:
                    shuffle = 0x0
                shuffle = check_random_state(shuffle)
                shuffle.shuffle(self.shuffle_mask)
        else:
            self.shuffle_mask = None

    def slice(self, n_slices=1):
        end = self.base_index + int(np.ceil(len(self.data) * n_slices / self.total))
        if self.shuffle_mask is not None:
            mask = self.shuffle_mask[self.base_index:end]
        else:
            mask = slice(self.base_index, end)

        ret = self.data[mask]
        self.base_index = end
        return ret

    def assign(self, worker_or_weight: Worker | float):
        if isinstance(worker_or_weight, Worker):
            worker_or_weight = worker_or_weight.get_weight()

        return self.slice(worker_or_weight)

    def iter(self, var_inp: WeightsType):
        for w in extract_weights(var_inp):
            yield self.assign(w)

    @property
    def finished(self):
        return self.base_index >= len(self.data)

    def inverse_shuffle(self, data):
        """从随机打乱的数据中恢复原索引

        Parameters
        ----------
        data
            乱序的数据，可能是分派任务后产出的数据

        Returns
        -------
        ordered_data
            还原为原顺序的数据
        """
        if self.shuffle_mask is None:
            return data
        else:
            inverse_mask = self.shuffle_mask.copy()
            inverse_mask[self.shuffle_mask] = np.arange(len(self.shuffle_mask))
            return np.asarray(data)[inverse_mask]

    def __getitem__(self, item):
        if item > 0:
            warnings.warn(f'Trying to access single allocator with index {item}, got self anyway.')
        return self


class MultiTaskAllocator(TaskAllocator):
    def __init__(self, *data_list, total=None, shuffle: ShuffleChoice = False):
        allocators = [SingleTaskAllocator(data, total=total, shuffle=shuffle) for data in data_list]
        self.allocators = Batch.of(allocators)

    def slice(self, n_slices=1):
        return list(self.allocators.slice(n_slices))

    def assign(self, worker_or_weight: Worker | float):
        return list(self.allocators.assign(worker_or_weight))

    def iter(self, var_inp: WeightsType):
        yield from zip(*self.allocators.iter(var_inp))

    @property
    def finished(self):
        return any(self.allocators.finished)

    def shuffle_masks(self):
        return [
            a.shuffle_mask for a in self.allocators
        ]

    def inverse_shuffle(self, *data_list):
        return [
            a.inverse_shuffle(d) for d, a in zip(data_list, self.allocators)
        ]

    def __getitem__(self, item):
        return self.allocators.batch_items[item]


class BoundedTaskAllocator(TaskAllocator):
    def __init__(self, allocator: TaskAllocator, weights: WeightsType):
        self.allocator = allocator
        self.weights = extract_weights(weights)

    @staticmethod
    def arrange(data, *, weights, shuffle: ShuffleChoice = False):
        weights = extract_weights(weights)
        return BoundedTaskAllocator(
            SingleTaskAllocator(data, total=sum(weights), shuffle=shuffle),
            weights=weights
        )

    @staticmethod
    def arrange_many(*data_list, weights, shuffle: ShuffleChoice = False):
        weights = extract_weights(weights)
        return BoundedTaskAllocator(
            MultiTaskAllocator(*data_list, total=sum(weights), shuffle=shuffle),
            weights=weights
        )

    def slice(self, n_slices=1):
        return self.allocator.slice(n_slices)

    def assign(self, worker: Worker | float):
        return self.allocator.assign(worker)

    def iter(self, var_inp: WeightsType):
        yield from self.allocator.iter(var_inp)

    @property
    def finished(self):
        return self.allocator.finished

    def inverse_shuffle(self, *data):
        return self.allocator.inverse_shuffle(*data)

    def __iter__(self):
        yield from self.allocator.iter(self.weights)

    def __getitem__(self, item):
        return self.allocator[item]


def extract_weights(weights: WeightsType):
    from .executor import Executor

    if isinstance(weights, Executor):
        weights = weights.get_workers()

    if isinstance(weights, Iterable):
        return [_extract_weight(w) for w in weights]

    return _extract_weight(weights)


def _extract_weight(weight: WeightsType):
    if isinstance(weight, Worker):
        return weight.get_weight()
    else:
        return weight  # 应该得是float，大概
