from .tile_locator import TileLocator
from ..constants import WebMercator


class SimpleTileLocator(TileLocator):
    def __init__(self, pattern: str, google_map_style=False):
        super().__init__()
        self.pattern = pattern
        self.google_map_style = google_map_style

    def query(self, col, row, level) -> str:
        if self.google_map_style:
            col, row, z = WebMercator.as_google_style_coord(col, row, level)
        return self.pattern.format(
            col=col, column=col, row=row, level=level,
            x=col, y=row, z=level,  # 允许缩写
        )
