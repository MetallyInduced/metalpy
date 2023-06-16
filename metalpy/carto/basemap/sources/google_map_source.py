import warnings

from .simple_tile_map_source import SimpleTileMapSource


class GoogleMapSource(SimpleTileMapSource):
    LangDefault = None
    LangChs = 'zh-CN'
    OffsetChs = 'cn'

    def __init__(self,
                 satellite=False,
                 labels=True,
                 arterial=None,
                 terrain=False,
                 labels_only=False,
                 terrain_only=False,
                 lang=None,
                 offset=None,
                 scale=None
                 ):
        """谷歌地图数据源

        Parameters
        ----------
        satellite
            是否使用卫星底图
        labels
            是否包含标签
        arterial
            是否包含主干道（不确定），对应选项 lyrs=m，只在带标签常规地图模式下有效，默认为True
            （not satellite and labels）
        terrain
            是否包含地形图，只在带标签常规地图模式下有效
            （not satellite and labels）
        labels_only
            下载只有标签的底图
        terrain_only
            下载只有地形图的底图
        lang
            地图渲染格式以及标注语言，即hl字段
        offset
            地图偏移，即gl字段，如果输入坐标为GCJ02坐标系则应该可以通过`offset=OffsetChs`校正
        """
        if terrain and arterial:
            warnings.warn('Terrain and arterial layer cannot exists at same time.')
        if terrain:
            if satellite or not labels:
                raise ValueError('Terrain layer supports only default map with labels.')
        if arterial:
            if satellite or not labels:
                warnings.warn('Arterial layer supports only default map with labels.')

        if arterial is None:
            arterial = True

        components = [r'http://mt[0123].google.com/vt/x=\{x\}&y=\{y\}&z=\{z\}&s=Gali']

        layers = 'm'
        if labels_only:
            layers = 'h'
        elif terrain_only:
            layers = 't'
        elif satellite:
            if labels:
                layers = 'y'
            else:
                layers = 's'
        elif not satellite:
            if labels:
                if terrain:
                    layers = 'p'
                elif arterial:
                    layers = 'm'
                else:
                    layers = 'r'
            else:
                layers = 't'

        components.append(f'lyrs={layers}')

        if lang is not None:
            components.append(f'hl={lang}')
        if offset is not None:
            components.append(f'gl={offset}')

        if scale is not None:
            components.append(f'scale={scale}')

        super().__init__(
            '&'.join(components),
            regex=True
        )
