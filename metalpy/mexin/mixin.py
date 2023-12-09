import functools
import warnings

from metalpy.utils.string import format_string_list


class Mixin:
    def __init__(self, this):
        """定义一个Mixin类，用于在其它类中添加新的功能
        接受MixinManager的管辖，MixinManager负责管理Mixin的生命周期与信息交换

        Note
        ----
            1. 通过MixinManager的add方法将Mixin添加到目标类中（通常依靠Patch来通知PatchContext执行）
            2. MixinManager会将其中所有方法的第二个参数绑定到目标类上，此时在目标类和mixin类中调用的方法都是已绑定了两个类的方法，形如

                def xxx(self, this, ...)，其中self是mixin对象实例，this是对应的目标类实例

            3. 可以通过MixinManager的get方法获取Mixin，例如

                mixinA = mixed_class_instance.mixins.get(MixinA)

            同时一些特殊方法会被排除: 1. 私有方法（即名字以__开头） 2. 魔术方法（即名字被__包围） 3. 属性方法
        """
        pass

    def post_apply(self, this):
        pass


def before(func=None, *, target=None):
    return TaggedMethod(func, tag=TaggedMethod.Before, target=target)


def after(func=None, *, keep_retval=False, target=None):
    return TaggedMethod(func, tag=TaggedMethod.After, keep_retval=keep_retval, target=target)


def replaces(func=None, *, keep_orig=False, target=None):
    return TaggedMethod(func, tag=TaggedMethod.Replaces, keep_orig=keep_orig, target=target)


class TaggedMethod:
    Before = 'before'
    After = 'after'
    Replaces = 'replaces'
    All = {Before, After, Replaces}

    def __init__(self, func, tag, target=None, keep_orig=False, keep_retval=False):
        self.func = func
        self.tag = tag
        self.target = target
        self.keep_orig = keep_orig
        self.keep_retval = keep_retval

    def __new__(cls, func, *args, **kwargs):
        if func is not None:
            return TaggedMethod(*args, **kwargs)(func)
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
