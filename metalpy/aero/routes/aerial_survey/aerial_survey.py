from __future__ import annotations

import warnings
from typing import TypeVar, TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import ArrayLike

from metalpy.carto.coords import Coordinates
from metalpy.utils import array_ops
from metalpy.utils.type import is_numeric_array, notify_package
from metalpy.aero.utils.line_analysis import split_lines, remove_aux_flights, analyze_lines

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
        diffs = np.diff(self.position, axis=0)
        dist = np.linalg.norm(diffs, axis=1)
        dist = dist[dist < 50]
        return dist.sum()

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

    def slice_lines(self, segments):
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
        """
        from . import AerialSurveyLines

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

        if auto_clear:
            aux_removed = remove_aux_flights(
                xs, ys,
                segments,
                stable_threshold=stable_direction_tol * 2
            )
        else:
            aux_removed = segments

        if merge:
            slopes, intercepts = analyze_lines(
                xs - np.percentile(xs, 50),
                ys - np.percentile(ys, 50),
                aux_removed
            )
            slope = np.mean(slopes)
            cos_theta = 1 / np.sqrt(1 + slope ** 2)

            di = abs(np.diff(intercepts)) * cos_theta
            criteria = np.percentile(di, 40) / 2

            merged_lines = [[0]]
            for i, merged in enumerate(di < criteria):
                if merged:
                    merged_lines[-1].append(i + 1)
                else:
                    merged_lines.append([i + 1])
        else:
            merged_lines = [[i] for i in range(len(aux_removed))]

        lines = self.slice_lines(aux_removed)
        lines = AerialSurveyLines([lines[line_ids].merge() for line_ids in merged_lines])

        return lines

    def plot(self, ax=None, show=True, **kwargs):
        from matplotlib import pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = None

        plt.plot(*np.asarray(self.position).T, **kwargs)

        if show and fig is not None:
            fig.show()

    def to_polydata(self, *, z=None, data_col: str | int = None, points_only=False):
        """导出航空数据为PyVista模型

        Parameters
        ----------
        z
            指定z值
        data_col
            指定作为z值的数据列，支持索引或列名
        points_only
            指示返回值是否只包含点坐标模型，不包含连接线

        Returns
        -------
        ret
            生成的PyVista模型
        """
        import pyvista as pv

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
        ret.verts = np.c_[np.ones(n_pts), range(n_pts)].ravel()

        if not points_only:
            ret.lines = np.r_[n_pts, range(n_pts)]

        return ret


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


def _guess_is_lon_lat_position(cols: Sequence[str], position):
    if 'lon' in cols[0].lower() and 'lat' in cols[1].lower():
        return True
    elif (
            all(abs(position.iloc[:, 0]) <= 180)  # 经度范围
            and all(abs(position.iloc[:, 1]) <= 90)  # 纬度范围
            and all(position.std() < 10)  # 如果单位为m，则主要区域范围不应小于10m，此时应为误用经纬度
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
