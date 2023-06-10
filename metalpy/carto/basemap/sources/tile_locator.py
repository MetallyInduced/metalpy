from abc import ABC, abstractmethod


class TileLocator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def query(self, col, row, level) -> str:
        pass
