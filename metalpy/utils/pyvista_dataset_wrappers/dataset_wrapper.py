from __future__ import annotations

from typing import TypeVar

from metalpy.utils.model import DataSetLike, DataSetLikeType

TDataSetWrapper = TypeVar('TDataSetWrapper', bound='DataSetWrapper')


class DataSetWrapper:
    def __init__(self, dataset: DataSetLike, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset

    def view(self, wrapper_type: type[TDataSetWrapper]) -> TDataSetWrapper:
        return wrapper_type(self.dataset)

    def wrap(self: TDataSetWrapper, dataset: DataSetLike) -> TDataSetWrapper:
        return type(self)(dataset)

    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def __getitem__(self, item):
        ret = self.dataset[item]
        if isinstance(ret, DataSetLikeType):
            return self.wrap(ret)
        else:
            return ret

    def __setitem__(self, item, val):
        if isinstance(val, DataSetWrapper):
            val = val.dataset
        self.dataset[item] = val

    def __repr__(self):
        ret = self.dataset.__repr__()
        return ret.replace(
            type(self.dataset).__name__,
            type(self).__name__,
        )
