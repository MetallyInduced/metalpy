from metalpy.mepa import Worker, Executor


class Distributable:
    def __init__(self):
        pass

    def should_distribute_to(self, worker: Worker):
        return True

    def distribute_to(self, executor: Executor, worker: Worker):
        if self.should_distribute_to(worker):
            return self
        else:
            return None
