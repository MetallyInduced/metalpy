import contextlib
import copy
import time


class Timer:
    def __init__(self):
        """单位为ns的计时器
        """
        self.started = 0
        self.elapsed = 0
        self.stopped = -1

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.check()

    def start(self):
        self.started = time.perf_counter_ns()

    def check(self):
        self.stopped = time.perf_counter_ns()
        self.elapsed = self.stopped - self.started

    @property
    def checked(self):
        return self.stopped != -1

    def __str__(self):
        if not self.checked:
            self.check()

        elapsed_mins, elapsed_secs, elapsed_milis, elapsed_micros = \
            perf_ns_epoch_time(self.elapsed)
        if elapsed_mins > 0:
            return f"{elapsed_mins} min {elapsed_secs} s"
        elif elapsed_secs > 0:
            return f"{elapsed_secs} s {elapsed_milis} ms"
        elif elapsed_milis > 0:
            return f"{elapsed_milis + elapsed_micros / 1000:.5f} ms"
        else:
            return f"{elapsed_micros:.5f} us"

    def __itruediv__(self, other):
        assert self.checked, 'Arithmetic operation is only available after timer being checked.'
        self.elapsed /= other
        return self

    def __truediv__(self, other):
        ret = copy.copy(self)
        ret /= other
        return ret


def epoch_time(elapsed_time):
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def perf_ns_epoch_time(elapsed_time):
    elapsed_micros = elapsed_time / 1000
    elapsed_milis = int(elapsed_micros / 1000)
    elapsed_secs = int(elapsed_milis / 1000)
    elapsed_mins = int(elapsed_secs / 60)

    elapsed_micros -= elapsed_milis * 1000
    elapsed_milis -= elapsed_secs * 1000
    elapsed_secs -= elapsed_mins * 60

    return elapsed_mins, elapsed_secs, elapsed_milis, elapsed_micros


@contextlib.contextmanager
def timed(msg='Elapsed time: ', callback=None, repeat=None):
    """构建计时块，对代码块进行计时

    Parameters
    ----------
    msg
        退出时输出的信息前缀
    callback
        退出时调用的回调函数，如果定义了 `callback` 则会忽略 `msg` 参数
    repeat
        手动定义重复次数，退出时自动将计时时间除以该次数

    Examples
    --------
    >>> with timed('Time elapsed: '):
    >>>     # do something
    >>>     print('finished!')
    >>>     pass
    <<< finished!
    <<< Time elapsed: 1s 234ms
    """
    t = Timer()
    try:
        t.start()
        yield
    finally:
        t.check()
        if repeat is not None:
            t /= repeat

        if callback is None:
            print(f"{msg}{t}")
        else:
            callback(t)


def repeated(repeat=20, msg='Elapsed time: ', callback=None):
    """构建循环计时块，对代码块进行计时

    Parameters
    ----------
    repeat
        手动定义重复次数，退出时自动将计时时间除以该次数
    msg
        退出时输出的信息前缀
    callback
        退出时调用的回调函数，如果定义了 `callback` 则会忽略 `msg` 参数

    Examples
    --------
    >>> for _ in repeated(20, 'Time elapsed: '):
    >>>     # do something
    >>>     pass
    <<< Time elapsed: 1s 234ms

    Notes
    -----
    TODO: 参照 `timeit` 模块的输出形式，提供更详细的运行时间信息，参照 `https://docs.python.org/3/library/timeit.html`
        例如："10000 loops, best of 5: 23.2 usec per loop"
    """
    with timed(msg, callback, repeat):
        for _ in range(repeat):
            yield
