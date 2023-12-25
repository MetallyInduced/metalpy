# 对应 taichi.lang.impl
from typing import TypeVar, overload, Any

import numpy as np

T = TypeVar('T')


@overload
def static(x: T) -> T: ...


@overload
def static(x: Any, *xs: Any) -> tuple: ...


def static(x, *xs):
    return x, *xs


def grouped(x):
    return np.ndindex(np.asarray(x).shape)
