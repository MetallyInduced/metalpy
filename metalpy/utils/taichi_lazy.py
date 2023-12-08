import functools


def lazy_kernel(fn):
    cache_key = '_kernel_cache'

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        kernel = getattr(fn, cache_key, None)
        if kernel is None:
            kernel = fn()
            setattr(fn, cache_key, kernel)

        return kernel(*args, **kwargs)

    return wrapper
