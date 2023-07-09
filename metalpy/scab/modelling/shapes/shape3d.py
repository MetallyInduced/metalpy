from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import Iterable

import numpy as np

from metalpy.utils.dhash import dhash
from metalpy.utils.bounds import Bounds, union
from ..transform import CompositeTransform, Transform, Translation, Rotation
from ..utils.mesh import is_inside_cuboid


class Shape3D(ABC):
    def __init__(self):
        self.transforms = CompositeTransform()

    @property
    def prune_mesh(self):
        """指示在子类do_place前自动根据Bounds框裁剪边界之外的网格
        """
        return True

    def place(self, mesh_cell_centers, progress):
        """计算模型体所占用的网格

        Parameters
        ----------
        mesh_cell_centers
            3维网格中心点坐标
        progress
            进度条实例或None，当子类的n_task属性不为1时，需自行更新进度条

        Returns
        -------
        ret
            布尔数组或浮点数组，浮点数组原则上来说取值范围应为[0, 1]，
            指示对应网格位置是否有效或有效的程度， 0 代表非活动网格
        """
        mesh_cell_centers = self.before_place(mesh_cell_centers, progress)

        indices = None
        if self.prune_mesh:
            p0, p1 = self.local_bounds.as_corners()
            indices = is_inside_cuboid(mesh_cell_centers, p0, p1 - p0)
            mesh_cell_centers = mesh_cell_centers[indices]

        ind = self.do_place(mesh_cell_centers, progress)

        if indices is not None:
            indices[indices] = ind
        else:
            indices = ind

        return indices

    def before_place(self, mesh_cell_centers, progress):
        return self.transforms.inverse_transform(mesh_cell_centers)

    @abstractmethod
    def do_place(self, mesh_cell_centers, progress):
        """计算模型体所占用的网格的实现函数，所有Shape3D的子类需要重写该函数

        Parameters
        ----------
        mesh_cell_centers
            3维网格中心点坐标
        progress
            进度条实例或None，当子类的n_task属性不为1时，需自行更新进度条

        Returns
        -------
        ret
            布尔数组或浮点数组，浮点数组原则上来说取值范围应为[0, 1]，
            指示对应网格位置是否有效或有效的程度，0代表非活动网格

        Notes
        -----
            注意：调用者会占用tqdm的position=0位置，如果继承的do_place方法内有自己定义的进度条，需要特别指定position为其他位置

            在Scene.build_mesh_worker中假定了0代表非活动网格，不参与模型间重叠部分的计算，请避免将0作为一个有意义的值输出
        """
        raise NotImplementedError()

    @abstractmethod
    def do_hash(self):
        """哈希函数，返回当前shape的哈希，
        由基类的__hash__调用，并混合transforms的哈希作为shape的完整哈希值返回

        Returns
        -------
        ret : int
            当前shape的哈希值
        """
        raise NotImplementedError()

    def __hash__(self):
        return dhash(self).digest()

    @abstractmethod
    def __dhash__(self):
        return dhash(self.transforms)

    def clone(self, deep=True):
        ret = self.do_clone(deep=deep)
        ret.transforms = self.transforms.clone()
        return ret

    @abstractmethod
    def do_clone(self, deep=True):
        raise NotImplementedError()

    def to_polydata(self):
        """将 Shape 转换为 PolyData 表示

        Returns
        -------
        ret
            PolyData 表示的 Shape

        Notes
        -----
            to_polydata 会在局部坐标系下的 PolyData 上应用变换，映射得到世界坐标系下的 PolyData

            因此继承类应当实现 to_local_polydata
        """
        import pyvista as pv

        ret: pv.PolyData = self.to_local_polydata()

        if ret is not None:
            if isinstance(ret, pv.MultiBlock):
                models = ret
            else:
                models = [ret]

            for model in models:
                pts = self.transforms.transform(model.points)
                model.points = pts

        return ret

    def to_local_polydata(self):
        """将 Shape 转换为本地坐标系下的 PolyData 表示

        Returns
        -------
        ret
            本地坐标系下 PolyData 表示的 Shape

        Notes
        -----
            继承类应当实现 to_local_polydata ，然后由 to_polydata 映射到世界坐标系中
        """
        raise NotImplementedError()

    @property
    def center(self) -> np.ndarray:
        """获取Shape在世界坐标系下的中心点坐标

        Returns
        -------
        ret : array(3)
            Shape在世界坐标系下的中心点坐标[x, y, z]
        """
        return self.transforms.transform(self.local_center)

    @property
    def bounds(self) -> Bounds:
        """获取Shape在世界坐标系下的长方体包围盒

        Returns
        -------
        ret : array(6)
            Shape在世界坐标系下的长方体包围盒[x0, x1, y0, y1, z0, z1]
        """
        bounds = self.oriented_bounds

        ret = np.zeros(6)
        ret[::2] = np.min(bounds, axis=1)
        ret[1::2] = np.max(bounds, axis=1)

        return Bounds(ret)

    @property
    def oriented_bounds(self) -> np.ndarray:
        """获取Shape在世界坐标系下的八点包围盒

        Returns
        -------
        ret : array(8, 3)
            Shape在世界坐标系下的八点包围盒[[x0, y0, z0], ...[x7, y7, z7]]
        """
        bounds = self.local_oriented_bounds
        return self.transforms.transform(bounds).T

    @property
    def local_center(self) -> np.ndarray:
        """获取Shape在局部坐标系下的中心点

        Returns
        -------
        ret : array(3)
            Shape在局部坐标系下的中心点坐标[x, y, z]
        """
        bounds = self.local_oriented_bounds
        return bounds.mean(axis=0)

    @property
    def local_bounds(self) -> Bounds:
        """获取Shape在局部坐标系下的长方体包围盒

        子类需要至少实现local_oriented_bounds和local_bounds中的一个

        Returns
        -------
        ret : array(6)
            Shape在局部坐标系下的长方体包围盒[x0, x1, y0, y1, z0, z1]
        """
        bounds = self.local_oriented_bounds
        ret = np.zeros(6)
        ret[::2] = np.min(bounds, axis=1)
        ret[1::2] = np.max(bounds, axis=1)

        return Bounds(ret)

    @property
    def local_oriented_bounds(self) -> np.ndarray:
        """获取Shape在局部坐标系下的八点包围盒

        子类需要至少实现local_oriented_bounds和local_bounds中的一个

        Returns
        -------
        ret : array(8, 3)
            Shape在局部坐标系下的八点包围盒[[x0, y0, z0], ...[x7, y7, z7]]
        """
        local_bounds = self.local_bounds

        nans = [np.nan, np.nan]
        xrng = local_bounds.xrange if local_bounds.n_axes > 0 else nans
        yrng = local_bounds.yrange if local_bounds.n_axes > 1 else nans
        zrng = local_bounds.zrange if local_bounds.n_axes > 2 else nans

        x, y, z = np.meshgrid(xrng, yrng, zrng, indexing='ij')
        return np.c_[x.ravel(), y.ravel(), z.ravel()]

    def apply(self, trans: Transform, inplace=False):
        """逻辑上对空间体位置进行变换，目前通过对网格点进行逆变换实现

        Parameters
        ----------
        trans
            待应用的变换
        inplace
            指示操作是否应用在当前实例上

        Returns
        -------
        ret
            当inplace为True，返回当前实例，否则返回一个变换后的新对象
        """
        if inplace:
            ret = self
        else:
            ret = self.clone()

        ret.transforms.add(trans)
        return ret

    def translate(self, x, y, z, inplace=False):
        """对Shape进行平移

        Parameters
        ----------
        x, y, z
            三个方向的位移量
        inplace
            指示操作是否应用在当前实例上

        Returns
        -------
        ret
            当inplace为True，返回当前实例，否则返回一个平移后的新对象
        """
        return self.apply(Translation(x, y, z), inplace=inplace)

    def rotate(self, y, a, b, degrees=False, seq='xyz', inplace=False):
        """对Shape进行旋转，方向遵循右手准则

        Parameters
        ----------
        seq
            旋转顺序
        y, a, b
            对应于seq的旋转量
        degrees
            指示y，a，b是否为角度制
        inplace
            指示操作是否应用在当前实例上

        Returns
        -------
        ret
            当inplace为True，返回当前实例，否则返回一个旋转后的新对象
        """
        return self.apply(Rotation(y, a, b, degrees=degrees, seq=seq), inplace=inplace)

    def translated(self, x, y, z, inplace=True):
        """对Shape进行平移

        Parameters
        ----------
        x, y, z
            三个方向的位移量
        inplace
            指示操作是否应用在当前实例上

        Returns
        -------
        ret
            默认应用与当前对象，并返回当前对象
        """
        return self.translate(x, y, z, inplace=inplace)

    def rotated(self, y, a, b, degrees=False, seq='xyz', inplace=True):
        """对Shape进行旋转，方向遵循右手准则

        Parameters
        ----------
        seq
            旋转顺序
        y, a, b
            对应于seq的旋转量
        degrees
            指示y，a，b是否为角度制
        inplace
            指示操作是否应用在当前实例上

        Returns
        -------
        ret
            默认应用与当前对象，并返回当前对象
        """
        return self.rotate(y, a, b, degrees=degrees, seq=seq, inplace=inplace)

    @property
    def volume(self):
        """获取该几何体的体积
        """
        raise NotImplementedError()

    @property
    def area(self):
        """获取该几何体的外表面积
        """
        raise NotImplementedError()

    @property
    def n_tasks(self):
        """指示子类的细分任务数。

        当返回值不为1时，子类可选择自行更新进度条，或重载progress_manually来指定是否自行更新进度条。

        Notes
        -----
        注意，如果n_tasks为动态值且可能为1，
        则可能需要手动重载progress_manually来协助Scene明确是否需要在Shape3D外进行更新
        """
        return 1

    @property
    def progress_manually(self):
        """指示子类是否自行更新进度条
        """
        return self.n_tasks != 1


def bounding_box_of(shapes: Iterable[Shape3D]):
    return reduce(union, (shape.bounds for shape in shapes))
