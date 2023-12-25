# 对应 taichi.lang._ndrange
import itertools
from typing import Sequence


def ndrange(*args):
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, Sequence):
            assert len(arg) == 2, (
                "Every argument of ndrange should be a scalar or a tuple/list like (begin, end)"
            )
            args[i] = (arg[0], max(arg[0], arg[1]))
        else:
            args[i] = (arg,)

    ranges = [range(*a) for a in args]
    return itertools.product(*ranges)
