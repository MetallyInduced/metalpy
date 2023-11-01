from __future__ import annotations

from typing import TypeVar

import pyvista as pv

from metalpy.utils.model import DataSetLike, DataSetLikeType, pv_ufunc_map

TDataSetWrapper = TypeVar('TDataSetWrapper', bound='DataSetWrapper')


class DataSetWrapper:
    def __init__(self, dataset: DataSetLike, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(dataset, DataSetWrapper):
            dataset = dataset.dataset

        self.dataset = dataset

    def view(self, wrapper_type: type[TDataSetWrapper]) -> TDataSetWrapper:
        return wrapper_type(self.dataset)

    def wrap(self: TDataSetWrapper, dataset: DataSetLike) -> TDataSetWrapper:
        return type(self)(dataset)

    def merge_all(self):
        # TODO: 清理 `merge` 后无用的贴图信息
        datasets = pv_ufunc_map(
            self.dataset,
            lambda dataset: dataset
        )

        return pv.merge(datasets)

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
        dataset_t = type(self.dataset).__name__
        this_t = type(self).__name__
        return ret.replace(
            dataset_t,
            f'{this_t}[{dataset_t}]',
        )
