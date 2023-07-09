from __future__ import annotations

import os
import traceback
import warnings
from functools import cache
from pathlib import Path
from typing import Union, TYPE_CHECKING

from .object_path import ObjectPath, objpath
from .type import ensure_as_iterable, undefined

if TYPE_CHECKING:
    from datetime import timedelta, datetime


PathLikeType = (str, os.PathLike)
PathLike = Union[str, os.PathLike]
cache_dir = Path('./.cache')


def ensure_dir(path: PathLike):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return os.fspath(path)


def ensure_filepath(path: PathLike):
    path = Path(path)
    ensure_dir(path.parent)
    return os.fspath(path)


def locate_file(path, *predicates, mode='first'):
    """快速查找特定的文件
    :param path: 查找路径
    :param predicates: 判断谓词
    :param mode: 返回模式
        all - 返回全部符合条件的文件
        first - 返回第一个符合条件的文件
        last - 返回最后一个符合条件的文件
    :return: mode为all时，返回一个符合条件的完整文件路径列表（可能为空），否则返回一个完整文件路径或None
    """
    immediate_return = True
    ret = []
    if mode == 'all':
        immediate_return = False

    iterable = os.listdir(path)
    if mode == 'last':
        iterable = reversed(iterable)

    for file in iterable:
        flag = True
        for predicate in predicates:
            flag = flag and predicate(file)
            if not flag:
                break

        if flag:
            ret.append(file)
            if immediate_return:
                break

    ret = [os.path.join(path, file) for file in ret]
    if immediate_return:
        return ret[0] if len(ret) > 0 else None
    else:
        return ret


def locate_prefixed_file(path, prefix):
    return locate_file(path, lambda file: file.startswith(prefix))


def locate_file_by(paths, *, prefix=None, suffix=None, ext=None, contains=None, filename=None, mode='all'):
    preds = []

    if prefix is not None:
        prefix = ensure_as_iterable(prefix, str)
        preds.append(lambda f: any(map(lambda s: f.startswith(s), prefix)))

    if suffix is not None:
        suffix = ensure_as_iterable(suffix, str)
        preds.append(lambda f: any(map(lambda s: f.endswith(s), suffix)))

    if ext is not None:
        ext = ensure_as_iterable(ext, str)
        preds.append(lambda f: os.path.splitext(f)[1] in ext)

    if contains is not None:
        contains = ensure_as_iterable(contains, str)
        preds.append(lambda f: any(map(lambda s: s in f, contains)))

    if filename is not None:
        filename = ensure_as_iterable(filename, str)
        preds.append(lambda f: f in filename)

    ret = []

    immediate_return = True
    if mode == 'all':
        immediate_return = False

    paths = ensure_as_iterable(paths, str)
    filepaths = None
    for path in paths:
        filepaths = locate_file(path, *preds, mode=mode)
        if immediate_return:
            if filepaths is not None:
                break
        else:
            ret.extend(filepaths)

    if immediate_return:
        return filepaths
    else:
        return ret


def locate_first_file_by(paths, *, prefix=None, suffix=None, ext=None, contains=None, filename=None):
    return locate_file_by(paths, prefix=prefix, suffix=suffix, ext=ext, contains=contains, filename=filename,
                          mode='first')


def locate_last_file_by(paths, *, prefix=None, suffix=None, ext=None, contains=None, filename=None):
    return locate_file_by(paths, prefix=prefix, suffix=suffix, ext=ext, contains=contains, filename=filename,
                          mode='last')


def locate_files_by(paths, *, prefix=None, suffix=None, ext=None, contains=None, filename=None):
    return locate_file_by(paths, prefix=prefix, suffix=suffix, ext=ext, contains=contains, filename=filename,
                          mode='all')


def git_ignore_directory(path):
    ignore_file = (Path(path) / '.gitignore')
    if not ignore_file.exists():
        ignore_file.write_text('# Created by metalpy automatically.\n*')


def make_cache_file(name, *args):
    return os.fspath(make_cache_file_path(name, *args))


def make_cache_directory(name, *args):
    return os.fspath(make_cache_directory_path(name, *args))


def make_cache_file_path(name, *args):
    ret = Path(cache_dir, name, *args).absolute()
    ensure_filepath(ret)
    git_ignore_directory(cache_dir)
    return ret


def make_cache_directory_path(name, *args):
    ret = Path(cache_dir, name, *args).absolute()
    ensure_dir(ret)
    git_ignore_directory(cache_dir)
    return ret


class FileCache:
    def __init__(self, func, ttl: 'timedelta' = None):
        self.func = func
        self.ttl = ttl

    def __call__(self, *args, **kwargs):
        return self.prepare()(*args, **kwargs)

    def force_update(self):
        return self.prepare(update=True)

    def prepare(self, update=False):
        return FileCachedCall(self, update=update)


class FileCachedCall:
    def __init__(self, cache_obj: FileCache, update=False):
        self.cache = cache_obj
        self.update = update

    @property
    def func(self):
        return self.cache.func

    @property
    def ttl(self):
        return self.cache.ttl

    @cache
    def get_func_namespace(self):
        return objpath(ObjectPath.of(self.func).nested_path)

    def __call__(self, *args, **kwargs):
        from metalpy.utils.dhash import dhash
        name = self.get_func_namespace()
        hashkey = dhash(self.func, *args, kwargs)

        val = undefined
        if not self.update:
            val = get_cache(hashkey, ttl=self.ttl, name=name)

        if val is not undefined:
            return val
        else:
            val = self.func(*args, **kwargs)
            put_cache(hashkey, val, name=name)
            return val


def get_cache_key(key, name: str | None = None):
    from metalpy.utils.dhash import dhash
    code = dhash(key).hexdigest(32)
    if name is not None:
        return f'{name}[{code}]'
    else:
        return code


def get_cache_path(key, name: str | None = None):
    return make_cache_directory_path('cached') / get_cache_key(key, name=name)


def put_cache(key, content, name: str | None = None):
    import cloudpickle

    file = get_cache_path(key, name=name)
    ensure_filepath(file)
    with file.open('wb') as f:
        cloudpickle.dump((key, content), f)


def get_cache(key, default=undefined, ttl: 'timedelta' = None, name: str | None = None):
    from datetime import datetime
    import cloudpickle

    file = get_cache_path(key, name=name)

    if file.exists():
        if ttl is not None:
            if datetime.now() - datetime.fromtimestamp(os.path.getmtime(file)) > ttl:
                return undefined

        with file.open('rb') as f:
            try:
                cache_key, content = cloudpickle.load(f)
                if cache_key == key:
                    return content
            except Exception:
                warnings.warn(traceback.format_exc() +
                              '\nException occurred when loading cache. Ignoring cached file.')

    return default


def clear_cache(key, name: str | None = None):
    file = get_cache_path(key, name=name)
    file.unlink(missing_ok=True)


def file_cached(func=None, ttl: 'timedelta' = None):
    import functools
    if func is None:
        return functools.partial(file_cached, ttl=ttl)
    else:
        return functools.wraps(func)(FileCache(func, ttl=ttl))


def openable(path):
    return isinstance(path, PathLikeType)
