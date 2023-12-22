import functools
import warnings

from metalpy.utils.string import format_string_list


class TaggedMethod:
    Before = 'before'
    After = 'after'
    Replaces = 'replaces'
    Ignore = 'ignore'
    All = {Before, After, Replaces, Ignore}

    def __init__(self, func, tag, target=None, keep_orig=False, keep_retval=False):
        self.func = func
        self.tag = tag
        self.target = target
        self.keep_orig = keep_orig
        self.keep_retval = keep_retval

    def __new__(cls, func, *args, **kwargs):
        if func is not None:
            return TaggedMethod(None, *args, **kwargs)(func)
        else:
            return super().__new__(cls)

    def __call__(self, func):
        self.func = func

        @functools.wraps(self.func)
        def wrapper(*args, **kwargs):
            return self.func(*args, **kwargs)

        wrapper.__tags__ = self
        return wrapper

    @staticmethod
    def verify(tag):
        if tag not in TaggedMethod.All:
            warnings.warn(f'Unknown mixin method type {tag},'
                          f' must be one of {format_string_list(TaggedMethod.All)}.'
                          f' Defaults to `{TaggedMethod.Replaces}`.')
            return TaggedMethod.Replaces
        return tag

    @staticmethod
    def check(meth):
        return getattr(meth, '__tags__', None)
