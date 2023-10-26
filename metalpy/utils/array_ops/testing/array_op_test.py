import numpy as np
import pandas as pd

from .. import is_array_like


def make_it_numpy(sth):
    if isinstance(sth, (pd.DataFrame, pd.Series)):
        return np.asarray(sth)
    else:
        return sth


class ArrayOpTest:
    def __init__(self, *args, expects=None, **kwargs):
        """构造针对ArrayOp的测试用例，
        只需要基于pandas目标提供参数，
        numpy目标下会自动将所有pandas目标进行转换后再执行测试

        Parameters
        ----------
        args, kwargs
            ArrayOp的参数
        expects
            期望的输出，如果给定为None，则会转换为pandas和numpy的等价性测试
            （测试两个目标下的输出是否一致，但不推荐）
        """
        self.args = args
        self.kwargs = kwargs
        self.expects = expects

    def test_pandas(self, op):
        got = op(*self.args, **self.kwargs)
        expected = self.expects

        if is_array_like(expected):
            return pd.testing.assert_frame_equal(
                got,
                expected
            )
        else:
            assert got == expected

    def test_numpy(self, op):
        args, kwargs = self.prepare_numpy()

        got = op(*args, **kwargs)
        expected = make_it_numpy(self.expects)

        if is_array_like(expected):
            return np.testing.assert_almost_equal(
                got,
                expected
            )
        else:
            assert got == expected

    def test_equivalence(self, op):
        args, kwargs = self.prepare_numpy()

        got = op(*args, **kwargs)
        expected = op(*self.args, **self.kwargs)
        expected = make_it_numpy(expected)

        if is_array_like(expected):
            return np.testing.assert_almost_equal(
                got,
                expected
            )
        else:
            assert got == expected

    def test(self, op):
        if self.expects is not None:
            self.test_pandas(op)
            self.test_numpy(op)
        else:
            self.test_equivalence(op)

    def prepare_numpy(self):
        args = [make_it_numpy(a) for a in self.args]
        kwargs = {k: make_it_numpy(v) for k, v in self.kwargs.items()}

        return args, kwargs
