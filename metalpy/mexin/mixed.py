import inspect
import warnings

from .injectors import extends, replaces, after, before
from .mixin import Mixin, TaggedMethod
from .patch import Patch
from .sentinal import NO_MIXIN


class MixinManager:
    """会被注册为目标类的mixins属性，由其他mixin调用进行mixin的注入或访问
    """
    def __init__(self, target):
        self._mixins = {}
        self.target = target

    def add(self, mixin_type, *args, **kwargs):
        """将mixin绑定到关联对象上

        Parameters
        ----------
        mixin_type
            mixin类
        args
            mixin类的位置参数
        kwargs
            mixin类的关键字参数

        Notes
        -----
            使用类加参数的形式来定义mixin的主要原因是可能存在mixin中需要持有一些对象的引用导致无法序列化，
            比如mixin对象内可能持有一个tqdm进度条实例，但如果再包一层参数代理类又会过于冗杂，因此采用这种方式定义mixin
        """
        target = self.target
        obj = mixin_type(target, *args, **kwargs)

        # 获取mixin的所有方法
        methods = inspect.getmembers(obj, predicate=MixinManager._is_method)
        for name, method in methods:
            if name.startswith('__') or \
               f'_{mixin_type.__name__}__' in name or \
               name in Mixin.__dict__:
                # 跳过1. 魔术方法 2. 私有方法 3. post_apply等mixin类的方法
                continue

            obj.__dict__[name] = self.bind_method(method, name=name)

        self._mixins[mixin_type] = obj
        obj.post_apply(target)

    def get(self, mixin_type, sentinel=False):
        if mixin_type in self._mixins:
            return self._mixins[mixin_type]
        else:
            if sentinel:
                return NO_MIXIN
            else:
                return None

    def get_mixin_or_sentinel(self, mixin_type):
        return self.get(mixin_type, sentinel=True)

    def bind_method(self, method, name=None):
        # TODO: 引入注解来标记是否需要替换，或者标记替换别的目标
        # TODO: 标记是否需要保留原函数
        tag = TaggedMethod.Replaces
        keep_orig = False
        if name is None:
            name = method.__name__

        mixing_config = TaggedMethod.check(method)
        if mixing_config:
            tag = TaggedMethod.verify(mixing_config.tag)
            keep_orig = mixing_config.keep_orig
            if mixing_config.target is not None:
                # 用户指定的替换目标，覆盖之前获取到的名字
                name = mixing_config.target

        target = self.target
        target_method = getattr(target, name, None)

        if not target_method:
            # Replaces模式下如果不存在目标函数则隐式变为Extend，扩展新方法
            if tag != TaggedMethod.Replaces:
                warnings.warn(f'Trying to perform `{tag}`'
                              f' on non-existing method `{name}`.'
                              f' Consider using `{TaggedMethod.Replaces}` instead.')

            extension = method
            if keep_orig:
                # Extend时本不存在原函数，但用户想keep_orig，所以给一个空函数来make user happy
                warnings.warn(f'Trying to keep non-existing original method `{name}`.'
                              f' Populating with an empty function.')

                def extension(*args, **kwargs): ...

            method = extends(target, name)(extension)

            if keep_orig:
                # 用于触发真正的Replaces
                target_method = method

        if target_method:
            # 如果在目标存在同名方法则替换、前后插入
            if tag == TaggedMethod.Replaces:
                method = replaces(target_method, nest=target, keep_orig=keep_orig)(method)
            elif tag == TaggedMethod.Before:
                method = before(target_method, nest=target)(method)
            elif tag == TaggedMethod.After:
                method = after(target_method, nest=target)(method)

        return method

    @staticmethod
    def _is_method(func):
        return (
            inspect.ismethod(func)
            or inspect.isfunction(func)
            or TaggedMethod.check(func)
        )


class Mixed(Patch):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.pre_applied_mixins = []

    def mix(self, mixin_type, *args, **kwargs):
        self.pre_applied_mixins.append((mixin_type, args, kwargs))
        return self

    def apply(self):
        self.add_injector(after(self.target.__init__), self.apply_mixins)

    def apply_mixins(self, this, *_, **__):
        manager = getattr(this, 'mixins', None)
        if manager is None:
            manager = MixinManager(this)
            this.mixins = manager
        for m, args, kwargs in self.pre_applied_mixins:
            manager.add(m, *args, **kwargs)

    @property
    def priority(self):
        return -1
