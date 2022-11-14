from abc import ABC, abstractmethod


class Transform(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, mesh):
        return mesh

    @abstractmethod
    def inverse_transform(self, mesh):
        return mesh

    @abstractmethod
    def clone(self):
        return Transform()

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()
