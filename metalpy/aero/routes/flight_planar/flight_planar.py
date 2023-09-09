from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, TYPE_CHECKING, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from metalpy.carto.coords import Coordinates
from metalpy.utils.file import PathLike

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


class FlightPlanar:
    Left = Rotation.from_euler('z', 90, degrees=True)
    Right = Rotation.from_euler('z', -90, degrees=True)

    def __init__(self, sample_distance: float = 10):
        """实现类tutle接口的航线规划功能，并提供航线动态图绘制接口

        Parameters
        ----------
        sample_distance
            航线的采样点间距，以下所有规划的航线上按该航距采样位置点
        """
        self.paths = []
        self.position = np.asarray([0, 0])
        self._direction = np.asarray([0, 1])
        self.sample_distance = sample_distance

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, v):
        self._direction = v / np.linalg.norm(v)

    def add_path(self, path):
        self.paths.append(np.asarray(path))

    def rotate_direction(self, angle, radians=False):
        self.direction = FlightPlanar.rot2d(
            Rotation.from_euler('z', angle, degrees=not radians),
            self.direction
        )

    def forward(self, distances):
        n_steps = int(distances / self.sample_distance)
        origin = self.position
        end = origin + self.direction * self.sample_distance * n_steps
        xs = np.linspace(origin[0], end[0], n_steps + 1)
        ys = np.linspace(origin[1], end[1], n_steps + 1)
        self.add_path(np.c_[xs, ys])

        real_end = origin + self.direction * distances
        self.position = real_end

    def turn(self, radius, theta, radians=False):
        if not radians:
            theta = np.deg2rad(theta)
        rot = FlightPlanar.get_rot(-theta)
        ortho_dir = FlightPlanar.rot2d(rot, self.direction)

        center = self.position + ortho_dir * radius

        a0 = FlightPlanar.vec2rad(ortho_dir)
        a1 = a0 + theta
        a_step = self.sample_distance / radius * np.sign(theta)
        angles = np.r_[np.arange(a0, a1, a_step), a1]

        current_angle = FlightPlanar.vec2rad(self.direction)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        path = np.c_[xs, ys]
        path += center
        path += self.position - path[0]

        self.add_path(path[:-1])
        self.position = path[-1]
        self.direction = FlightPlanar.rad2vec(current_angle + theta)

    def stay(self, n_steps):
        path = self.position[np.newaxis, :].repeat(n_steps, axis=0)
        self.add_path(path)

    def spiral(self, r0, theta, round_increment, radians=False, return_radix=False):
        """等距螺旋线

        .. math::
            r(\\theta) = k\\theta + r_0

        Parameters
        ----------
        r0
            起始极径
        theta
            螺旋线旋转角（720°为2个周期，半径增加2 * `round_increment`）
        round_increment
            每周期极径增量 / 线距（k = `round_increment` / (2 * np.pi) ）
        radians
            传入值是否为弧度
        return_radix
            是否返回每一段的弧长，若为False则无返回值
        """
        if not radians:
            theta = np.deg2rad(theta)
        rot = FlightPlanar.get_rot(-theta)
        ortho_dir = FlightPlanar.rot2d(rot, self.direction)

        center = self.position + ortho_dir * r0
        a = 0
        inc = round_increment / np.pi / 2
        angles = []
        total = abs(theta)

        radix = None
        if return_radix:
            radix = []

        while a < total:
            radius_0 = r0 + a * inc
            const = FlightPlanar._spiral_arc_length(radius_0, inc) + self.sample_distance
            da = self.sample_distance / radius_0
            radius_x = radius_0 + da * inc

            # 牛顿法求解给定弧长下的等距螺旋线的极角增量  Rx+1 = Rx - F(Rx) / F'(Rx)
            radius_x = (
                    radius_x -
                    (FlightPlanar._spiral_arc_length(radius_x, inc) - const)
                    / FlightPlanar._spiral_arc_length_derive(radius_x, inc)
            )

            da = (radius_x - radius_0) / inc
            a = a + da
            angles.append(a)
            if radix is not None:
                radix.append((radius_0 + radius_x))

        angles = np.asarray(angles + [abs(theta)])
        radix = angles * inc + r0
        xs = radix * np.cos(angles) * np.sign(theta)
        ys = radix * np.sin(angles)

        current_angle = FlightPlanar.vec2rad(self.direction)
        path = np.c_[xs, ys]
        path = FlightPlanar.rot2d(Rotation.from_euler('z', current_angle - np.pi / 2), path)
        path += center
        path += self.position - path[0]

        self.add_path(path[:-1])
        self.position = path[-1]
        self.direction = FlightPlanar.rad2vec(current_angle + theta)

        if return_radix:
            return radix

    def cut_last_path(self, *, by_distance=None, by_steps=None):
        """切割上一条航段为两个航段

        Parameters
        ----------
        by_distance
            按指定长度序列依次切分航段，每个航段长度等同于数组每个元素
        by_steps
            按时间步数分割为若干航段，每个航段的时间步数等同于数组每个元素

        Notes
        -----
            若给定长度不够切分完整航段，则多余的部分单独成为一个航段；
            若总长度超出航段长度，则切完即停止，后续指定的长度会忽略，切割出的最后一个航段长度可能会小于指定长度；
            航段长度尽可能接近指定长度，但总是小于等于指定长度；
            如果指定单独的值，则视为长度为1的数组。
        """
        path = self.paths.pop(-1)
        if by_distance is not None:
            if not isinstance(by_distance, Iterable):
                by_distance = (by_distance,)

            by_steps = []
            for dist in by_distance:
                by_steps.append(dist // self.sample_distance)

        if not isinstance(by_steps, Iterable):
            by_steps = (by_steps,)

        idx = 0
        for n_steps in by_steps:
            end_idx = idx + n_steps
            self.paths.extend(path[idx:end_idx])
            idx = end_idx

    def split_last_path(self, *, by_distance=None, by_steps=None):
        """切割上一条航段

        :param by_distance: 按长度分割为若干航段
        :param by_steps: 按时间步数分割为若干航段
        """
        path = self.paths.pop(-1)
        if by_distance is not None:
            n_segments = len(path) * self.sample_distance // by_distance
        else:
            n_segments = len(path) // by_steps
        self.paths.extend(np.array_split(path, n_segments))

    def build(self):
        return np.concatenate(self.paths)

    def plot2d(self,
               screenshot: bool | PathLike = False,
               off_screen=False,
               arrows: bool | int = True,
               color: bool | 'Colormap' = False,
               alpha: float | Sequence[float] = 1,
               ax=None
               ):
        import matplotlib.pyplot as plt

        route = self.build()
        min_xy = np.min(route, axis=0)
        max_xy = np.max(route, axis=0)
        extent = max_xy - min_xy
        half = extent.max() / 2 * 1.1
        center = min_xy + extent / 2

        # TODO: change fig_size according to extent
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = None
            ax = ax

        if not color:
            ax.plot(*route.T, zorder=0)
        else:
            if color is True:
                cmap = plt.get_cmap('tab10')
            else:
                cmap = color

            if not isinstance(alpha, Sequence):
                alpha = [alpha]

            n_colors = len(cmap.colors)
            n_alphas = len(alpha)
            for i, path_i in enumerate(self.paths):
                ax.plot(
                    *path_i.T,
                    zorder=0,
                    color=cmap(i % n_colors),
                    alpha=alpha[i % n_alphas]
                )

        if arrows:
            n_pts = route.shape[0]
            if arrows is True:
                arrows = min(max(n_pts - 3, 1), 5)
            plot_arrows(ax, route, n_arrows=arrows, width=half * 0.01, zorder=1)

        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)

        if fig:
            if screenshot:
                if screenshot is True:
                    import time
                    screenshot = Path(f'./{time.time_ns()}.png').absolute()
                else:
                    screenshot = Path(screenshot).absolute()
                fig.savefig(screenshot, bbox_inches='tight')

            if not off_screen:
                fig.show()

            plt.close(fig)

    def plot2d_segments(self, path, **kwargs):
        for _ in self.plot2d_segments_customized(path=path, **kwargs):
            pass

    def plot2d_segments_customized(
            self,
            path,
            *,
            cmap=None,
            inactive_color=(0.7, 0.7, 0.7),
            inactive_alpha=0.3,
            show_index=True,
            fig=None
    ):
        from PIL import Image
        import tqdm
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        if cmap is None:
            cmap = plt.get_cmap('tab10')

        if fig is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            ax = fig.subplots()

        n_paths = len(self.paths)
        n_ratio = int(np.ceil(n_paths / len(cmap.colors)))
        colors = cmap.colors * n_ratio
        cmap_colors = [inactive_color] * n_paths
        alphas = [inactive_alpha] * n_paths

        pics = []
        for i in tqdm.trange(n_paths):
            yield ax  # 用于配置axis
            cmap_colors[i] = colors[i]
            alphas[i] = 1
            self.plot2d(ax=ax, color=ListedColormap(cmap_colors), alpha=alphas)

            if show_index:
                ax.text(*Coordinates(self.paths[i]).bounds.center, str(i), fontsize=10)

            fig.canvas.draw()
            size = fig.canvas.get_width_height() * np.asarray(fig.get_dpi()) / 100
            pics.append(Image.frombuffer("RGB", tuple(size.astype(int)), fig.canvas.tostring_rgb()))
            ax.cla()
            cmap_colors[i] = inactive_color
            alphas[i] = inactive_alpha

        pics[0].save(
            path,
            save_all=True,
            append_images=pics[1:],
            duration=500,
            loop=0
        )

    def plot2d_traversal(
        self,
        path,
        *,
        fig=None,
        frame_rate=24,
        total_time: int | float = 2,
    ):
        from PIL import Image
        from matplotlib import pyplot as plt
        import tqdm

        if fig is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            ax = fig.add_subplot()

        route = self.build()
        min_xy = np.min(route, axis=0)
        max_xy = np.max(route, axis=0)
        extent = max_xy - min_xy
        half = extent.max() / 2 * 1.1
        center = min_xy + extent / 2

        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)

        ax.plot(*route.T, zorder=1)

        n_time_steps = len(route)
        n_frames = int(total_time * frame_rate)
        group_size = int(np.ceil(n_time_steps / n_frames))
        head_length = int(n_time_steps * 0.1)  # 用于限定着色位置为航线头部，避免航线较长时颜色混乱

        pics = []
        for i in tqdm.trange(0, n_time_steps, group_size, desc='Creating animation...'):
            rows = route[:i+group_size]
            colors = np.arange(-len(rows), 0)
            colors[:-head_length] = -head_length
            pr = ax.scatter(*rows.T, c=colors)
            fig.canvas.draw()
            size = fig.canvas.get_width_height() * np.asarray(fig.get_dpi()) / 100
            pics.append(Image.frombuffer("RGB", tuple(size.astype(int)), fig.canvas.tostring_rgb()))

            pr.remove()

        pics.extend([pics[-1]] * 5)  # 在结果上停留

        pics[0].save(
            path,
            save_all=True,
            append_images=pics[1:],
            duration=1000 / frame_rate,
            loop=0
        )

    @staticmethod
    def get_rot(theta):
        if theta >= 0:
            return FlightPlanar.Left
        else:
            return FlightPlanar.Right

    @staticmethod
    def rot2d(rot, vec2d, inverse=False):
        if vec2d.ndim == 1:
            return rot.apply(np.r_[vec2d, 0], inverse=inverse)[:2]
        else:
            return rot.apply(np.c_[vec2d, np.zeros(len(vec2d))], inverse=inverse)[:, :2]

    @staticmethod
    def vec2rad(vec):
        return math.atan2(vec[1], vec[0])

    @staticmethod
    def rad2vec(rad):
        return np.asarray((np.cos(rad), np.sin(rad)))

    @staticmethod
    def _spiral_arc_length(r, k):
        """等距螺旋线弧长不定积分

        .. math::
            \\int (\\sqrt{r(\\theta)^2 + r'(\\theta)^2}) d(\\theta)

            = \\int (\\sqrt{r^2 + k^2} / k) dr

            = (r \\sqrt{k^2 + r^2} + k^2 \\log(\\sqrt{k^2 + r^2} + r)/(2k) + constant
        """
        k2 = k ** 2
        p = np.sqrt(k2 + r ** 2)
        return (r * p + k2 * np.log(p + r)) / (2 * k)

    @staticmethod
    def _spiral_arc_length_derive(r, k):
        """等距螺旋线弧长不定积分的导数，即待积式

        .. math::
            \\sqrt{r^2 + k^2} / k
        """
        return np.sqrt(k ** 2 + r ** 2) / k


def plot_arrows(ax, path, n_arrows=None, width=10, cmap=None, **kwargs):
    n_pts = path.shape[0]
    if n_arrows is None:
        n_arrows = min(max(n_pts - 3, 1), 5)
    arrow_locs = np.linspace(n_pts * 0.02, n_pts * 0.98, n_arrows).astype(int)

    if cmap is None:
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['red'] + ['black'] * (n_arrows - 1))
    else:
        if cmap.N != n_arrows:
            cmap = cmap._resample(n_arrows)

    first_color = 'red'
    normal_color = 'black'
    color = first_color

    for i, idx in enumerate(arrow_locs):
        start = path[idx]
        end = path[idx + 1]
        ax.arrow(*start, *(end - start),
                 width=width,
                 facecolor=cmap(i),
                 edgecolor=cmap(i),
                 **kwargs)

        if color == first_color:
            color = normal_color
