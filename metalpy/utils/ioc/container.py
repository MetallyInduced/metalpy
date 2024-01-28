from __future__ import annotations

from typing import Any, TypeVar, overload, Callable

from .service import ServiceDescriptor

ServiceT = TypeVar('ServiceT')


class ServiceContainer:
    def __init__(self):
        self._collection: dict[type | ServiceDescriptor, Any] = {}
        self._lazy_collection: dict[type | ServiceDescriptor, Callable] = {}

    @overload
    def __getitem__(self, service_class: type[ServiceT]) -> ServiceT: ...

    @overload
    def __getitem__(self, service_descriptor: ServiceDescriptor[ServiceT]) -> ServiceT: ...

    def __getitem__(self, varinp):
        ret = self._collection.get(varinp, None)

        if ret is None:
            provider = self._lazy_collection.get(varinp, None)
            if provider is not None:
                ret = self.add_singleton(varinp, self._build(provider))

        if ret is None:
            ret = self.add_singleton(varinp, self.build(varinp))

        return ret

    @overload
    def build(self, service_class: type[ServiceT]) -> ServiceT: ...

    @overload
    def build(self, service_descriptor: ServiceDescriptor[ServiceT]) -> ServiceT: ...

    def build(self, varinp):
        provider = self._get_provider(varinp)
        ret = self._build(provider)

        return ret

    def add_singleton(self, varinp: type | ServiceDescriptor, service=None, lazy=True):
        if service is not None:
            # 分别指定服务描述符和服务实例
            ret = self._collection[varinp] = service
        else:
            if isinstance(varinp, (type, ServiceDescriptor)):
                # `varinp` 为服务描述符
                provider = self._get_provider(varinp)

                ret = None
                if lazy:
                    # 构造缓式服务
                    self._lazy_collection[varinp] = provider
                else:
                    # 显式指定非缓式，直接构造服务实例
                    ret = self._collection[type(varinp)] = self._build(provider)
            else:
                # `varinp` 为服务实例，将其类型视为服务描述符
                ret = self._collection[type(varinp)] = varinp

        return ret

    def _get_provider(self, varinp: type | ServiceDescriptor):
        provider = getattr(varinp, '__service__', None)

        if provider is None:
            if isinstance(varinp, type):
                provider = varinp  # 构造函数也可以作为service provider

        if provider is None:
            raise ValueError(
                f'No service found for `{repr(varinp)}`.'
            )

        return provider

    def _build(self, provider: Callable):
        ret = provider(self)  # TODO: Build service by type annotations of parameter
        return ret
