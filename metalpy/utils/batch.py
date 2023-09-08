from typing import TypeVar, Sequence, Generic

T = TypeVar('T')


class Batch(Generic[T]):
    __slots__ = ['_items']

    def __init__(self, items: Sequence[T]):
        """用于批量操作对象

        Parameters
        ----------
        items
            需要执行批量操作的对象集合
        """
        self._items = items

    @staticmethod
    def of(items: Sequence[T]) -> "T | Batch[T]":
        """创建一个批操作对象

        Parameters
        ----------
        items
            需要执行批量操作的对象集合

        Returns
        -------
        batch
            批操作对象

        Notes
        -----
        返回值和直接创建一模一样，但是依靠指定返回值类型来欺骗IDE给出提示
        """
        return Batch(items)

    @property
    def batch_items(self):
        return self._items

    def __len__(self):
        return len(self.batch_items)

    def __iter__(self):
        yield from self.batch_items

    def __call__(self, *args, **kwargs):
        return Batch([t.__call__(*args, **kwargs) for t in self])

    def __getitem__(self, item):
        return Batch([t.__getitem__(item) for t in self])

    def __getattr__(self, item):
        if item not in Batch.__slots__:
            return Batch([getattr(t, item) for t in self])
        else:
            return getattr(self, item)

    def __setattr__(self, item, val):
        if item not in Batch.__slots__:
            for t in self:
                setattr(t, item, val)
        else:
            super().__setattr__(item, val)
