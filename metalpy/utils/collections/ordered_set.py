import copy
from collections import abc
from typing import TypeVar, Mapping, Collection

_T = TypeVar('_T')


class OrderedSet(dict[_T, None], abc.MutableSet[_T]):
    """ modified from https://stackoverflow.com/a/1653978
    """
    items = None
    values = None

    def __new__(cls, iterable=None):
        if iterable is None:
            return super().__new__(cls)
        else:
            return cls.fromkeys(iterable)

    def __init__(self, *_):
        super().__init__()

    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            if not isinstance(s, Mapping):
                s = dict.fromkeys(s)
            super().update(s)

    def add(self, elem: _T):
        self[elem] = None

    def discard(self, elem: _T):
        self.pop(elem, None)

    def isdisjoint(self, other):
        for k in other:
            if k in self:
                return True
        else:
            return False

    def __ior__(self, other):
        self.update(other)
        return self

    def __or__(self, other):
        ret = copy.copy(self)
        ret.__ior__(other)

        return ret

    def __and__(self, other):
        return type(self)([
            k for k in other if k in self
        ])

    def __iand__(self, other):
        ret = self.__and__(other)
        self.clear()
        self.update(ret)

        return self

    def __sub__(self, other):
        return type(self)([
            k for k in other if k not in self
        ])

    def __isub__(self, other):
        ret = self.__sub__(other)
        self.clear()
        self.update(ret)

        return self

    def __xor__(self, other):
        if not isinstance(other, Collection):
            other = list(other)  # 保证可重入

        return type(self)(
            [k for k in self if k not in other]
            + [k for k in other if k not in self]
        )

    def __ixor__(self, other):
        ret = self.__xor__(other)
        self.clear()
        self.update(ret)

        return self

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self)!r})"

    def __str__(self) -> str:
        return f"{{{list(self)!r}}}"

    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)
    union = property(lambda self: self.__or__)
