import abc
from typing import Generic, TypeVar, TYPE_CHECKING

from metalpy.utils.abcd import ABCD

if TYPE_CHECKING:
    from .container import ServiceContainer

ServiceT = TypeVar('ServiceT')


class ServiceDescriptor(Generic[ServiceT], ABCD):
    @abc.abstractmethod
    def __service__(self, data: 'ServiceContainer') -> ServiceT:
        ...

    @abc.abstractmethod
    def __hash__(self):
        ...

    @abc.abstractmethod
    def __eq__(self, other):
        ...


class Service(ABCD):
    @classmethod
    @abc.abstractmethod
    def __service__(cls: type[ServiceT], data: 'ServiceContainer') -> ServiceT:
        ...
