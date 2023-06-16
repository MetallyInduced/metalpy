from .tile_locator import TileLocator
from ..constants import WebMercator


class SimpleTileLocator(TileLocator):
    def __init__(self, pattern: str, bottom_left_as_origin=False):
        super().__init__()
        self.pattern = pattern
        self.bottom_left_as_origin = bottom_left_as_origin

    def query(self, col, row, level) -> str:
        col, row = WebMercator.warp_tile_coord(col, row, level, self.bottom_left_as_origin)
        return self.pattern.format(
            col=col, column=col, row=row, level=level,
            x=col, y=row, z=level,  # 允许缩写
        )
