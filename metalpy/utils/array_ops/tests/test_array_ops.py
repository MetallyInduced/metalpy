import pandas as pd

from ..testing import ArrayOpTest


def make_indexed_dataframes(*arrs, columns=None):
    if columns is None:
        columns = range(0, len(arrs))
    for arr, col in zip(arrs, columns):
        yield pd.DataFrame(arr, columns=[col])


def make_dataframes(*arrs, indexed=False):
    if indexed:
        columns = None
    else:
        columns = [0] * len(arrs)
    yield from make_indexed_dataframes(*arrs, columns=columns)


def test_cstack():
    from metalpy.utils.array_ops import cstack
    ArrayOpTest(
        *make_indexed_dataframes([1, 2, 3], [4, 5, 6]),
        expects=pd.DataFrame([[1, 4], [2, 5], [3, 6]])
    ).test(cstack)


def test_rstack():
    from metalpy.utils.array_ops import rstack
    ArrayOpTest(
        *make_dataframes([1, 2, 3], [4, 5, 6]),
        expects=pd.DataFrame([1, 2, 3, 4, 5, 6])
    ).test(rstack)
