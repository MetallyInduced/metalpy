from __future__ import annotations

from abc import ABC, abstractmethod
from typing import overload


class WorkerContext(ABC):
    """实现消息通信上下文，通过序列化传递到Worker中，Worker调用fire给指定的event事件发送信息
    """

    @abstractmethod
    def fire(self, event, *args, **kwargs):
        pass

    @overload
    def bind(self, event) -> BoundWorkerContext: ...

    @overload
    def bind(self, event, idx: int | None) -> IndexedBoundWorkerContext: ...

    def bind(self, event, idx: int | None = None) -> BoundWorkerContext | IndexedBoundWorkerContext:
        if idx is None:
            return BoundWorkerContext(self, event)
        else:
            return IndexedBoundWorkerContext(self, event, idx=idx)


class BoundWorkerContext:
    def __init__(self, context: WorkerContext, event):
        """支持将WorkerContext绑定到给定事件上，从而在调用fire函数时不需要额外指定事件名

        Parameters
        ----------
        context
            事件消息通信上下文
        event
            绑定的事件名
        """
        self.context = context
        self.event = event

    def fire(self, *args, **kwargs):
        self.context.fire(self.event, *args, **kwargs)


class IndexedBoundWorkerContext(BoundWorkerContext):
    def __init__(self, context: WorkerContext, event, idx: int | None = None):
        """支持将WorkerContext绑定到给定事件上，从而在调用fire函数时不需要额外指定事件名

        Parameters
        ----------
        context
            事件消息通信上下文
        event
            绑定的事件名
        event
            绑定的事件回调函数的下标
        """
        super().__init__(context, event)
        self.idx = idx

    def __getstate__(self):
        """idx不需要一起传过去
        """
        return {
            'context': self.context,
            'event': self.event
        }
