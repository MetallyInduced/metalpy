from __future__ import annotations

import warnings

from metalpy.mexin import Mixin
from metalpy.mexin.utils import TypeMap
from metalpy.utils.object_path import get_full_qualified_path


class DispatcherMixin(Mixin):
    """抽象根据 `Mixin` 的宿主类匹配并添加具体实现 `Mixin` 的插件范式

    主要通过三个功能函数实现：
        1. DispatcherMixin.implements: 用于为特定类指定实现 `Mixin`
        2. DispatcherMixin.find_impl: 用于根据宿主类查找实现 `Mixin`
        3. DispatcherMixin.post_apply: 织入宿主类后，调用 `find_impl` ，查找通过 `implements` 注册的实现 `Mixin` 并完成第二次织入
    """
    _impls: TypeMap | None = None
    allow_match_parent = False
    warns_when_not_matched = False

    def __init_subclass__(cls, *, allow_match_parent=False, warns_when_not_matched=False):
        cls._impls = TypeMap(allow_match_parent=allow_match_parent)
        cls.allow_match_parent = allow_match_parent
        cls.warns_when_not_matched = warns_when_not_matched

    def __init__(self, this, *args, **kwargs):
        super().__init__(this)
        self.args = args
        self.kwargs = kwargs

    @classmethod
    @Mixin.ignores
    def find_impl(cls, typ: type):
        """根据宿主类查找对应的实现 `Mixin`

        Parameters
        ----------
        typ
            宿主类型
        """
        impl = cls._impls.get(typ)

        if impl is None and cls.warns_when_not_matched:
            warnings.warn(
                f'`{cls.__name__.strip("_")}` support for {get_full_qualified_path(typ)} is not implemented.'
                f' Ignoring it.'
            )

        return impl

    @classmethod
    @Mixin.ignores
    def implements(cls, target: type | str):
        """为指定的目标类设置实现 `Mixin`

        Parameters
        ----------
        target
            目标类，支持直接指定类型，也支持通过类路径指定
        """
        def decorator(func):
            cls._impls.map(target, func)
            return func

        return decorator

    def post_apply(self, this):
        impl = self.find_impl(type(this))

        if impl is not None:
            this.mixins.add(impl, *self.args, **self.kwargs)
