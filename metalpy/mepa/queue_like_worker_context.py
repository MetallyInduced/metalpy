from .worker_context import WorkerContext


class QueueLikeWorkerContext(WorkerContext):
    def __init__(self, queue=None):
        self.queue = queue

    def fire(self, event, *args, **kwargs):
        self.queue.put((event, (args, kwargs)))
