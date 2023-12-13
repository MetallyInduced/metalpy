import itertools
from typing import Any

from metalpy.utils.arg_specs import ArgSpecs
from metalpy.utils.type import undefined


class LazyClassFactory:
    def __init__(self, cls, *args, _argspecs: ArgSpecs = None, **kwargs):
        """用于实现类的延迟构造，
        通过construct可以使用对应的类进行构造，new_kwargs指定用于替换或追加的参数，
        通过clone可以复制一个新的延迟构造器

        Parameters
        ----------
        cls
            待构造目标类
        _argspecs
            预先生成的参数信息
        args
            位置参数
        kwargs
            关键字参数
        """
        self.cls = cls
        if _argspecs is None:
            self.argspecs = ArgSpecs.of(cls)
        else:
            self.argspecs = _argspecs

        self.argspecs.push_args(*args)
        self.argspecs.bind_kwargs(**kwargs)

    def find_param(self, predicate, remove=False, repl=None, default=undefined):
        """在参数列表中搜索指定参数

        Parameters
        ----------
        predicate
            搜索判断条件
        remove
            指示是否移除搜索到的参数
        repl
            替换项，如果搜索到结果且repl不为None则用该参数替换

        Returns
        -------
        ret
            搜索到的目标参数
        """
        ret = default
        matched = False

        for i, val in self.argspecs.bound_posargs.items():
            argspec = self.argspecs.posarg_specs[i]
            matched = predicate(i, val)

            if not matched:
                matched = matched or predicate(argspec.name, val)

            if matched:
                ret = val
                if repl is not None:
                    self.argspecs.bind_arg(argspec, repl)
                elif remove:
                    self.argspecs.remove(argspec)
                break

        if not matched:
            for argspec, val in self.argspecs.bound_kwargs.items():
                matched = predicate(argspec.name, val)

                if matched:
                    ret = val
                    if repl is not None:
                        self.argspecs.bind_arg(argspec, repl)
                    elif remove:
                        self.argspecs.remove(argspec)
                    break

        if undefined == ret:
            raise ValueError('Bound arg not found.')

        return ret

    def find_param_by_type(self, type, remove=False, repl=None):
        """在参数列表中搜索指定类型的参数

        Parameters
        ----------
        type
            指定的参数类型
        remove
            指示是否移除搜索到的参数
        repl
            替换项，如果搜索到结果且repl不为None则用该参数替换

        Returns
        -------
        ret
            搜索到的目标参数
        """
        try:
            return self.find_param(lambda k, v: isinstance(v, type), remove=remove, repl=repl)
        except ValueError:
            raise ValueError(f'Bound arg with type `{type}` not found.')

    def find_param_by_name(self, name, remove=False, repl=None):
        """在参数列表中搜索指定名字的参数

        Parameters
        ----------
        name
            指定的参数名
        remove
            指示是否移除搜索到的参数
        repl
            替换项，如果搜索到结果且repl不为None则用该参数替换

        Returns
        -------
        ret
            搜索到的目标参数
        """
        try:
            return self.find_param(lambda k, v: k == name, remove=remove, repl=repl)
        except ValueError:
            raise ValueError(f'Bound arg with name `{name}` not found.')

    def construct(self, replace_by_type: dict[type, Any] = None, **new_kwargs):
        """构造类对象

        Parameters
        ----------
        replace_by_type
            需要按类型替换的参数列表
        new_kwargs
            新的关键字参数

        Returns
        -------
        ret
            构造的类对象
        """
        temp_args = self.argspecs.clone()

        if replace_by_type is None:
            replace_by_type = {}

        if new_kwargs is None:
            new_kwargs = {}

        for argspec, val in itertools.chain(
                list(zip(temp_args.posarg_specs, temp_args.bound_posargs)),
                list(temp_args.bound_kwargs.items())
        ):
            name = argspec.name
            orig = val
            t = type(val)
            if t in replace_by_type:
                val = replace_by_type[t]
                t = type(val)

            if argspec.name in new_kwargs:
                val = new_kwargs.pop(name)
                t = type(val)

            if t == LazyClassFactory:
                val = val.construct()

            if val is not orig:
                temp_args.bind_arg(argspec, val)

        if len(new_kwargs) > 0:
            for k, v in new_kwargs.items():
                temp_args.bind_arg(k, v)

        return temp_args.call(self.cls)

    def clone(self):
        return LazyClassFactory(self.cls, _argspecs=self.argspecs.clone())

    def get(self, name_or_index, default=undefined):
        return self.argspecs.get_bound_arg(name_or_index, default=default)

    def pop(self, name_or_index, default=undefined):
        return self.argspecs.pop_bound_arg(name_or_index, default=default)

    def __getitem__(self, key):
        return self.argspecs.get_bound_arg(key)

    def __setitem__(self, key, value):
        return self.argspecs.bind_arg(key, value)

    def __delitem__(self, key):
        return self.argspecs.remove(key)
