from __future__ import annotations

import copy
import warnings
from typing import TypeVar, TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import ArrayLike

from metalpy.aero.utils.line_analysis import split_lines
from metalpy.carto.coords import Coordinates
from metalpy.utils import array_ops
from metalpy.utils.array_ops import get_column
from metalpy.utils.matplotlib import check_axis
from metalpy.utils.pandas import drop_by_indices
from metalpy.utils.type import is_numeric_array, notify_package

if TYPE_CHECKING:
    import pandas as pd
    from .aerial_survey_lines import AerialSurveyLines


TAerialSurvey = TypeVar('TAerialSurvey', bound='AerialSurvey')


class AerialSurvey:
    def __init__(self, position, data=None):
        """构造航空探测数据对象

        Parameters
        ----------
        position
            坐标信息
        data
            附属的数据
        """
        self.position = position
        self.data = data

    @classmethod
    def from_array(
            cls: type[TAerialSurvey],
            df: ArrayLike | 'pd.DataFrame',
            pos_cols: Sequence[str | int] = None,
            force_lon_lat=False
    ) -> TAerialSurvey:
        if pos_cols is None:
            pos_cols = _infer_position_columns(df)
        else:
            assert len(pos_cols) == 2, '`pos_cols` should clarify exactly two columns to be horizontal position.'

        position = array_ops.get_column(df, pos_cols)

        if _guess_is_lon_lat_position(pos_cols, position) and not force_lon_lat:
            warnings.warn('Position based on longitude-latitude coordinates may interfere the line analysis process.'
                          ' Transformed to UTM local system with unit `meter`.')
            position = _wrap_lon_lat_to_utm(position=position)
            data = df
        else:
            position = array_ops.get_column(df, pos_cols)
            data = array_ops.get_columns_except(df, pos_cols)

        return cls(position, data)

    @staticmethod
    def from_photo_exifs(paths):
        import pandas as pd

        from metalpy.aero.utils.exif import load_gps_info_from_exifs

        gps = load_gps_info_from_exifs(paths)
        pos = gps[['Longitude', 'Latitude']]

        utm = _wrap_lon_lat_to_utm(position=pos)
        gps = pd.concat([utm, gps], axis=1)

        return AerialSurvey.from_array(gps)

    @classmethod
    def concat(cls: type[TAerialSurvey], lines) -> TAerialSurvey:
        pos = array_ops.concat([line.position for line in lines])
        data = [line.data for line in lines]
        if any((d is None for d in data)):
            data = None
        else:
            data = array_ops.concat(data)

        return cls(pos, data)

    @property
    def center(self):
        return Coordinates(self.position).bounds.center.tolist()

    @property
    def length(self):
        dist = self.point_distances
        dist = dist[dist < 50]
        return dist.sum()

    @property
    def point_distances(self):
        diffs = np.diff(self.position, axis=0)
        dist = np.linalg.norm(diffs, axis=1)
        return dist

    @property
    def area(self):
        try:
            from cv2 import minAreaRect as min_area_rect
        except ImportError:
            notify_package(
                pkg_name='cv2',
                reason='`cv2.minAreaRect` is needed for more precise area computation.',
                install='pip install opencv-python'
            )
            from metalpy.utils.algos import min_area_rect_pca as min_area_rect

        rect_center, rect_size, rect_rotation = min_area_rect(
            self.position.to_numpy().astype(np.float32)
        )

        origin_area = np.prod(Coordinates(self.position).bounds.extent)
        oriented_area = np.prod(rect_size)

        return min(origin_area, oriented_area)

    def slice_data(self, slice0: slice | tuple, *slices: slice | tuple):
        if self.data is None:
            return None
        else:
            slices = [slice0, *slices]
            ret = []
            for s in slices:
                if not isinstance(s, slice):
                    s = slice(*s)
                ret.append(self.data[s])

            return array_ops.concat(ret)

    def slice_pos(self, slice0: slice | tuple, *slices: slice | tuple):
        slices = [slice0, *slices]
        ret = []
        for s in slices:
            if not isinstance(s, slice):
                s = slice(*s)
            ret.append(self.position[s])

        return array_ops.concat(ret)

    def slice_line(self, slice0: slice | tuple, *slices: slice | tuple):
        from . import AerialSurveyLine

        return AerialSurveyLine(
            self.slice_pos(slice0, *slices),
            self.slice_data(slice0, *slices),
        )

    def slice_lines(self, segments) -> 'AerialSurveyLines':
        from . import AerialSurveyLines

        return AerialSurveyLines([self.slice_line(seg) for seg in segments])

    def extract_lines(
            self,
            auto_clear=True,
            window_size=None,
            window_length=10,
            stable_window_size=None,
            stable_window_length=50,
            stable_direction_tol: float = 3,
            merge=True
    ) -> 'AerialSurveyLines':
        """分割提取测区中的测线

        Parameters
        ----------
        auto_clear
            是否自动清理航向角与主航向角差距较大的航段（通常为辅助航段，例如起降的航段）。
            如果包含多方向飞行的航线则不应该启用此选项。
        window_size, window_length
            用于分析航向的窗口测点数。应当大于飞机产生明显位移的窗口大小。
            过小时可能会导致测线划分对飞机飞行扰动敏感产生错误划分，过大时可能导致转弯等非平稳飞行测点也被容纳进测线。
        stable_window_size
            用于确定为航线的窗口测点数。应当大于转弯等非平稳飞行过程的窗口大小。
            过小时可能导致转弯等非平稳测点被统计为测线，过大时可能导致某些较短测线被意外移除。
        window_length, stable_window_length
            对应的窗口长度。若不指定对应size而指定length，则自动统计测点间距然后通过length求取窗口测点数。
        stable_direction_tol
            用于判断是否归属于相同测线的航向容差，单位（度）
        merge
            指示是否在自动清理辅助航段后（`auto_clear=True`）再次合并相邻航线。
            通常用于分多次完成的大测区测量，中间涉及多次起飞降落的辅助航段，移除后重新合并为完整测区的航线。

        Returns
        -------
        lines
            分析提取的各条测线

        Notes
        -----
        航段整体上可以分为三类：工作区间，辅助区间和非平稳区间
            1. 工作区间为正常平稳飞行，测量数据使用的飞行区间

            2. 辅助区间为平稳飞行的区间，但并不进行数据测量，例如飞机起飞降落、转向等动作

            3. 非平稳区间，飞行状态切换之间的状态，例如高速转向、悬停等状态下，会被判定为非平稳区间

        `extract_lines` 实现了排除 `非平稳区间` ，并提取切分 `工作区间` 和 `辅助区间` ，
        然后通过 `remove_auxiliary_lines` 判断并移除 `辅助区间` 。
        """
        xs, ys = np.asarray(self.position).T
        segments = split_lines(
            xs, ys,
            window_size=window_size,
            window_length=window_length,
            stable_window_size=stable_window_size,
            stable_window_length=stable_window_length,
            stable_direction_tol=stable_direction_tol
        )

        assert len(segments) > 0, 'Failed to pre-analyze flight lines by heading direction.'

        lines = self.slice_lines(segments)
        if auto_clear:
            lines.remove_auxiliary(inplace=True)
        if merge:
            lines.merge_neighbors(inplace=True)

        return lines

    def trim_auxiliary(
            self,
            keep_takeoff_and_landing=False,
            interval_threshold=None,
            ref_lines: 'AerialSurveyLines' | None = None,
            aux_lines: 'AerialSurveyLines' | None = None,
            inplace=False
    ) -> 'AerialSurvey':
        """基于`extract_lines`的测线划分结果，从原始测线数据中清理掉辅助轨迹，从而合并为一个单航次的数据

        例如对于较大测区，需要分多航次进行测量时，清除掉中途的多次起飞降落的轨迹，结果可以视为一个航次内完成采集的数据

        区别于`extract_lines`，转向过程等常规航次应有的轨迹会保留

        Parameters
        ----------
        keep_takeoff_and_landing
            指示是否保留起飞和降落轨迹，为False则只保留工作状态下的航线轨迹
        interval_threshold
            跳变点判定阈值，若不指定，则取为航线间距
        ref_lines
            用作参考的工作航线划分，若不指定，则使用默认参数进行航线划分
        aux_lines
            用作参考的辅助航线划分，若不指定，则使用默认参数进行航线划分
        inplace
            指示是否就地操作，默认为False，即返回修改后的实例，若否，则直接修改当前实例

        Returns
        -------
        trimmed
            移除掉辅助轨迹后的Survey实例

        Notes
        -----
        如果外部给定`ref_lines`参数，则注意需要指定`auto_clear`和`merge`为False来保留识别到的辅助航段信息

        >>> survey = AerialSurvey.from_array(...)
        >>> lines = survey.extract_lines(
        >>>     auto_clear=False,
        >>>     merge=False
        >>>     stable_direction_tol=2.5
        >>> )
        >>> survey.trim_auxiliary(ref_lines=lines)
        """
        import pandas as pd

        assert hasattr(self.data, 'index'), \
            ('`remove_auxiliary` requires indexed arrays,'
             ' try using pd.DataFrame as array type.'
             '\n'
             '    AerialSurvey.from_array(pd.DataFrame(your_data))'
             '\n')

        if ref_lines is None:
            ref_lines = self.extract_lines(
                auto_clear=False,
                merge=False,
            )
            aux_lines = ref_lines.pop_auxiliary()
            ref_lines.merge_neighbors(inplace=True)
        else:
            ref_lines = copy.deepcopy(ref_lines)

        specs = ref_lines.get_line_specs()
        line_interval = np.max([s.line_interval for s in specs if not np.isnan(s.line_interval)])
        aux_indices = []

        if aux_lines is not None:
            # 排除已经判定为辅助航线的部分
            # 注意，拐弯部分也有可能会被判定为辅助航线，因此判断和`line_interval`航线间隔相差过大的为辅助航线而非拐弯
            for line in aux_lines:
                if max(line.position.index) <= min(ref_lines[0].position.index):
                    continue  # 第一次起飞轨迹会在后方单独讨论
                if min(line.position.index) >= max(ref_lines[-1].position.index):
                    continue  # 最后一次降落轨迹会在后方单独讨论
                if not (0.5 <= line.length / line_interval <= 2):
                    aux_indices.append(line.position.index)

        if not keep_takeoff_and_landing:
            # 第一条测线之前和最后一条测线之后，分别对于第一次起飞和最后一次降落的轨迹
            aux_indices.extend([
                pd.RangeIndex(0, min(ref_lines[0].position.index)),
                pd.RangeIndex(max(ref_lines[-1].position.index), max(self.data.index) + 1),
            ])

        for line in ref_lines:
            # 所有已经判定为工作航线的区间，如果并非连续的数据点，则中间间断的部分也可能是辅助航线
            ref = line.position.index
            if not isinstance(ref, pd.RangeIndex):
                interval_pos = np.where(np.diff(ref) > 1)[0]
                for ip in interval_pos:
                    aux_indices.append(pd.RangeIndex(ref[ip] + 1, ref[ip + 1]))

        dropped_pos = drop_by_indices(self.position, aux_indices)
        dropped_flight = AerialSurvey.from_array(dropped_pos)
        pos = dropped_flight.position

        # 检查移除相关航线后导致留下的间断航段
        dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        normal = np.percentile(dist, 50)
        if interval_threshold is None:
            # 不给定间断阈值的话，采用航线间距作为判定阈值
            interval_threshold = line_interval

        threshold_pos = np.where(dist - normal > interval_threshold)[0]
        threshold_pos = np.r_[-1, threshold_pos, len(pos) - 1]
        windows = np.lib.stride_tricks.sliding_window_view(threshold_pos, 2) + 1
        segments = AerialSurvey(pos).slice_lines(windows)

        # 依据间断距离分割的各个大航段，如果存在工作航线属于该航段，则保留
        # 若不存在，则可能为之前未清理干净的辅助轨迹，需要移除
        idling_indices = []
        i = 0
        for segment in segments:
            contains_line = False

            seg_index = segment.position.index
            smin, smax = min(seg_index), max(seg_index)

            while i < len(ref_lines):
                ref = ref_lines[i]
                ref_ind = ref.position.index
                if min(ref_ind) < smin:
                    # 工作航线在航段之前
                    i += 1
                elif max(ref_ind) < smax:
                    # 工作航线包含在航段之中
                    contains_line = True
                    i += 1
                    break
                else:
                    # 工作航线在航段之后
                    break

            if not contains_line:
                idling_indices.append(seg_index)

        dropped_pos = drop_by_indices(dropped_pos, idling_indices)
        if self.data is not None:
            dropped_data = drop_by_indices(self.data, aux_indices + idling_indices)
        else:
            dropped_data = None

        if inplace:
            self.position = dropped_pos
            self.data = dropped_data
            ret = self
        else:
            ret = AerialSurvey(dropped_pos, dropped_data)

        return ret

    def plot(self, ax=None, show=True, **kwargs):
        with check_axis(ax, show=show, figsize=(5, 5)) as ax:
            ax.plot(*np.asarray(self.position).T, **kwargs)

    def scatter(self, ax=None, show=True, color=True, **kwargs):
        with check_axis(ax, show=show, figsize=(5, 5)) as ax:
            if 's' not in kwargs:
                kwargs['s'] = 0.05  # 默认的点尺寸太大，会互相重叠

            if color and 'c' not in kwargs:
                kwargs['c'] = range(len(self.position))

            ax.scatter(*np.asarray(self.position).T, **kwargs)

    def to_polydata(
            self,
            *,
            z=None,
            data_col: str | int = None,
            points_only=False,
            color=True
    ):
        """导出航空数据为PyVista模型

        Parameters
        ----------
        z
            指定z值，或者通过字符串指定z值所在列。
            如果给定为单个数值类型，则同时用作所有数据点的高度值。
        data_col
            指定作为z值的数据列，支持索引或列名
        points_only
            指示返回值是否只包含点坐标模型，不包含连接线
        color
            指示返回值是否包含顺序信息，默认为True，则PointData内绑定数据下标，渲染时自动着色

        Returns
        -------
        ret
            生成的PyVista模型
        """
        import pyvista as pv

        if isinstance(z, str):
            data_col = z
            z = None

        if data_col is not None:
            z = array_ops.get_column(self.data, data_col)
        else:
            if z is None:
                z = 0
            if not is_numeric_array(z):
                z = np.full(len(self.position), z)

        pos = np.c_[self.position, z]
        n_pts = pos.shape[0]

        ret = pv.PolyData(pos)

        if not points_only:
            ret.lines = np.r_[n_pts, range(n_pts)]

        if color:
            ret.point_data['index'] = range(n_pts)

        return ret

    def to_planar(self):
        from metalpy.aero.routes import FlightPlanar
        planar = FlightPlanar(np.percentile(self.point_distances, 50))
        planar.add_path(self.position)

        return planar

    def __len__(self):
        return len(self.position)


POSITION_KEYWORDS = [
    ['X', 'Y'],
    ['Longitude', 'Latitude'],
    ['Long', 'Lat'],
    ['Lon', 'Lat'],
]
POSITION_KEYWORDS = [
    (proc(xk), proc(yk))
    for xk, yk in POSITION_KEYWORDS
    for proc in [
        lambda k: k.capitalize(),
        lambda k: k.upper(),
        lambda k: k.lower(),
    ]
][1:]


def _infer_position_columns(arr):
    if hasattr(arr, 'keys'):
        def _infer():
            keys: set[str] = set(arr.keys())
            for xkey, ykey in POSITION_KEYWORDS:
                if xkey in keys and ykey in keys:
                    return [xkey, ykey]

                for possible_xkey in keys:
                    segs = possible_xkey.split(xkey)
                    n_segs = len(segs)
                    if n_segs <= 1:
                        continue

                    for i in range(1, n_segs):
                        left = xkey.join(segs[:i])
                        right = xkey.join(segs[i+1:])
                        inferred_ykey = f'{left}{ykey}{right}'

                        if inferred_ykey in keys:
                            return [possible_xkey, inferred_ykey]

            return None

        keys_found = _infer()
    else:
        keys_found = None

    if keys_found is None:
        keys_found = [0, 1]

    return keys_found


def _guess_is_lon_lat_position(cols: Sequence[str | int], position):
    if isinstance(cols[0], str) and 'lon' in cols[0].lower() and 'lat' in cols[1].lower():
        return True
    else:
        x, y = get_column(position, 0), get_column(position, 1)
        if (
                all(abs(x) <= 180)  # 经度范围
                and all(abs(y) <= 90)  # 纬度范围
                and all(np.std(x) < 10)  # 如果单位为m，则主要区域范围不应小于10m，此时应为误用经纬度
        ):
            return True
        else:
            return False


WrapLonLatToUTM = array_ops.OperatorDispatcher('WrapLonLatToUTM')


def _wrap_lon_lat_to_utm(position):
    return WrapLonLatToUTM.dispatch(type(position))(position)


@WrapLonLatToUTM.register(array_ops.ArrayType.numpy)
def _wrap_lon_lat_to_utm_np(position):
    coords = Coordinates(position, crs=Coordinates.WGS_84).warp(query=Coordinates.SearchUTM)
    return coords.to_numpy()


@WrapLonLatToUTM.register(array_ops.ArrayType.pandas)
def _wrap_lon_lat_to_utm_pd(position):
    import pandas as pd
    coords = Coordinates(position, crs=Coordinates.WGS_84).warp(query=Coordinates.SearchUTM)
    return pd.DataFrame(coords, columns=['UTMX', 'UTMY'], index=position.index)
