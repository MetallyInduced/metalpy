from abc import ABC, abstractmethod


class WorkerContext(ABC):
    """实现消息通信上下文，通过序列化传递到Worker中，Worker调用fire给指定的event事件发送信息
    """

    @abstractmethod
    def fire(self, event, *args, **kwargs):
        pass

    def bind(self, event):
        return BoundWorkerContext(self, event)


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
