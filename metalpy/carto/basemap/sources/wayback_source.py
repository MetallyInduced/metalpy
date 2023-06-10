import warnings

import numpy as np

from .simple_tile_map_source import SimpleTileMapSource


class WaybackSource(SimpleTileMapSource):
    def __init__(self):
        """Wayback()地图数据源

        Parameters
        ----------
        """
        super().__init__(
            '&'.join(components),
            google_map_style=True,
            regex=True
        )
