from __future__ import annotations

from typing import Iterable

from metalpy.utils.regex_pattern import RegexPattern
from .tile_map_source import TileMapSource
from .simple_tile_locator import SimpleTileLocator


class SimpleTileMapSource(TileMapSource):
    def __init__(self, pattern: str | Iterable[str], bottom_left_as_origin=False, regex=False):
        """简单Tile服务器地图源

        Parameters
        ----------
        pattern
            地图源，一般为一个格式化URL，
            例如`https://domain.com/<path>/{z}{x}{y}.png`
            或者`https://domain.com/<path>/{level}/{col}/{row}.png`。
        bottom_left_as_origin
            指示是否以左下方为原点（标准WMTS约定左上方为原点）
        regex
            指示pattern串是否为有穷正则表达式，若为True，会使用RegexPattern将其展开为各个子域名

        Notes
        -----
        若启动regex，{level}, {row}, {y}这些tile坐标占位符会被错误识别为正则的限定次数重复语法，此时需要使用 `\\\\{` 进行转义

        References
        ----------
        metalpy.utils.regex_pattern.RegexPattern: 有穷正则表达式展开算法
        """
        super().__init__()
        self.pattern = pattern
        self.regex = regex
        self.bottom_left_as_origin = bottom_left_as_origin

    def __iter__(self) -> Iterable[SimpleTileLocator]:
        if isinstance(self.pattern, str):
            if self.regex:
                patterns = RegexPattern(self.pattern)
            else:
                patterns = (self.pattern,)
        else:
            patterns = self.pattern

        for pat in patterns:
            yield SimpleTileLocator(pat, bottom_left_as_origin=self.bottom_left_as_origin)
