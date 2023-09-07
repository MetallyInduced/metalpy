from __future__ import annotations

import itertools
import warnings

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from metalpy.utils.ear_clip import ear_clip


class TurtlePath:
    def __init__(self, pts=None, fill=False):
        if pts is None:
            pts = []
        self.pts = pts
        self.fill = fill

    def append(self, p):
        self.pts.append(p)

    @property
    def empty(self):
        return len(self.pts) == 0

    @property
    def available(self):
        if self.fill:
            return self.n_points >= 3
        else:
            return self.n_points >= 2

    @property
    def n_points(self):
        return len(self.pts)


class TurtleFillingContext:
    def __init__(self, turtle: PyVistaTurtle):
        self.turtle = turtle

    def __enter__(self):
        self.turtle.begin_fill()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.turtle.end_fill()

    def __bool__(self):
        return self.turtle.path and self.turtle.path.fill


class PyVistaTurtle:
    def __init__(self):
        self.current_position = np.asarray([0, 0, 0])
        self.current_orientation = np.asarray([0, 1, 0])

        self.path: TurtlePath | None = None
        self.shapes: list[TurtlePath] = []

        self.pendown()

    def forward(self, distance):
        self.goto(self.position() + self.heading() * distance)

    fd = forward

    def backward(self, distance):
        self.forward(-distance)

    back = bk = backward

    def left(self, angle, radians=False):
        self.rotate('z', angle, radians=radians)

    lt = left

    def right(self, angle, radians=False):
        self.rotate('z', -angle, radians=radians)

    rt = right

    def goto(self, pos):
        pos = np.asarray(pos)
        if self.isdown():
            if self.path.empty:
                self.path.append(self.position())
            self.path.append(pos)
        self.current_position = pos

    setpos = setposition = goto

    def set_axis(self, axis, val):
        if isinstance(axis, str):
            axis = ord(axis) - ord('x')
        now_pos = self.position()
        now_pos[axis] = val

        self.goto(now_pos)

    def setx(self, x):
        self.set_axis(0, x)

    def sety(self, y):
        self.set_axis(1, y)

    def setz(self, z):
        self.set_axis(2, z)

    def setheading(self, heading):
        heading = np.asarray(heading)
        self.current_orientation = heading / np.linalg.norm(heading)

    seth = setheading

    def home(self):
        self.goto((0, 0, 0))
        self.seth((1, 0, 0))

    def circle(self, radius):
        raise NotImplemented()

    def rotate(self, axes, angle, radians=False):
        angle = np.atleast_1d(angle)
        r = Rotation.from_euler(axes, angle, degrees=not radians)
        self.setheading(r.apply(self.heading())[0])

    def position(self):
        return self.current_position

    pos = position

    def towards(self):
        raise NotImplemented()

    def xcor(self):
        return self.position()[0]

    def ycor(self):
        return self.position()[1]

    def zcor(self):
        return self.position()[2]

    def heading(self):
        return self.current_orientation

    def pendown(self):
        if self.path is None:
            self.path = TurtlePath(fill=False)

    pd = down = pendown

    def penup(self):
        if self.path is not None:
            if not self.path.empty:
                if self.path.n_points < 2:
                    warnings.warn('Only one point in a path. This should not happen. Ignoring it.')
                elif self.path.fill and self.path.n_points < 3:
                    warnings.warn('Insufficient vertices to fill, ignoring it.')
                else:
                    self.shapes.append(self.path)
            self.path = None

    pu = up = penup

    def isdown(self):
        return self.path is not None

    def filling(self):
        """区别于经典turtle API，PyVistaTurtle通过filling函数额外提供Python上下文管理器风格的filling方式。

        Examples
        --------
        >>> turtle = PyVistaTurtle()
        >>> if turtle.filling():  # 判断是否在filling模式，上下文对象隐式转换为bool值
        >>>     print('Filling??')
        >>> with turtle.filling():  # 通过with语法进入filling上下文
        >>>     if turtle.filling():
        >>>         print('Filling!')
        """
        return TurtleFillingContext(self)

    def begin_fill(self):
        self.penup()
        self.pendown()
        self.path.fill = True

    def end_fill(self):
        self.penup()
        self.pendown()

    def to_polydata(self):
        points = []
        lines = []
        cells = []

        additional_shapes = []
        if self.path.available:
            additional_shapes.append([self.path])

        base_point_id = 0
        for s in itertools.chain(self.shapes, *additional_shapes):
            points.append(s.pts)
            if not s.fill:
                pids = np.arange(base_point_id, base_point_id + s.n_points - 1)
                lines.append(np.c_[np.full_like(pids, 2), pids, pids + 1])
            else:
                triangles = ear_clip(s.pts) + base_point_id
                cells.append(np.c_[np.full(*triangles.shape), triangles])

            base_point_id += s.n_points

        kwargs = {}

        points = np.vstack(points)
        if len(lines) > 0:
            lines = np.vstack(lines)
            kwargs['lines'] = lines
        if len(cells) > 0:
            cells = np.vstack(cells)
            kwargs['cells'] = cells

        return pv.PolyData(
            points,
            **kwargs
        )
