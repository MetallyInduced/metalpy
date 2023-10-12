from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import Iterable, TYPE_CHECKING

import numpy as np
import tqdm

from metalpy.utils.dhash import dhash
from metalpy.utils.bounds import Bounds, union
from ..transform import CompositeTransform, Transform, Translation, Rotation
from ..utils.mesh import is_inside_bounds

if TYPE_CHECKING:
    from metalpy.scab.modelling import Scene


class Shape3D(ABC):
    def __init__(self):
        self.transforms = CompositeTransform()

    @property
    def prune_mesh(self):
        """指示在子类do_place前自动根据Bounds框裁剪边界之外的网格
        """
        return True

    def place(self, mesh_cell_centers, progress=False):
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

        Notes
        -----
        内部同时负责对坐标点进行变换并进行裁剪，保证`do_place`内部只需要按局部坐标系计算
        """
        progress = self._check_progress(progress)

        mesh_cell_centers = self.before_place(mesh_cell_centers, progress)

        indices = None
        if self.prune_mesh:
            indices = is_inside_bounds(mesh_cell_centers, self.local_bounds)
            mesh_cell_centers = mesh_cell_centers[indices]

        ind = self.do_place(mesh_cell_centers, progress)

        if indices is not None:
            indices[indices] = ind
        else:
            indices = ind

        return indices

    def compute_implicit_distance(self, mesh_cell_centers, progress=False):
        """计算模型体到空间中任一点的有向距离，且小于0代表在模型内

        Parameters
        ----------
        mesh_cell_centers
            3维网格中心点坐标
        progress
            进度条实例或None，当子类的n_task属性不为1时，需自行更新进度条

        Returns
        -------
        ret
            浮点数组，指示空间中任意一点到模型体表面的有向距离，且小于0代表在模型内

        Notes
        -----
        内部同时负责对坐标点进行变换，保证`do_compute_implicit_distance`内部只需要按局部坐标系计算
        """
        progress = self._check_progress(progress)

        mesh_cell_centers = self.before_place(mesh_cell_centers, progress)
        return self.do_compute_implicit_distance(mesh_cell_centers, progress)

    def before_place(self, mesh_cell_centers, progress):
        return self.transforms.inverse_transform(mesh_cell_centers)

    def do_place(self, mesh_cell_centers, progress):
        """计算模型体所占用的网格的实现函数，所有Shape3D的子类应当重写该函数

        默认实现依赖to_local_polydata()，基于PyVista的select_enclosed_points，存在性能问题

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
        在Scene.build_mesh_worker中假定了0代表非活动网格，不参与模型间重叠部分的计算，请避免将0作为一个有意义的值输出

        外部`place`通过`before_place`对坐标点进行了逆变换，只需要按局部坐标系计算即可
        """
        import pyvista as pv

        mesh = pv.PolyData(mesh_cell_centers).cast_to_unstructured_grid()
        surface = self.to_local_polydata().extract_surface()

        selection = mesh.select_enclosed_points(surface, tolerance=0.0, check_surface=True)

        if progress and self.progress_manually:
            progress.update(self.n_tasks)

        return selection['SelectedPoints'].view(bool, np.ndarray)

    def do_compute_implicit_distance(self, mesh_cell_centers, progress):
        """计算模型体到空间中任一点的有向距离的具体实现

        默认实现依赖to_local_polydata()，基于PyVista的compute_implicit_distance，
        需要进一步优化可以重载此函数

        Parameters
        ----------
        mesh_cell_centers
            3维网格中心点坐标
        progress
            进度条实例或None，当子类的n_task属性不为1时，需自行更新进度条

        Returns
        -------
        ret
            浮点数组，指示空间中任意一点到模型体表面的有向距离，且小于0代表在模型内

        Notes
        -----
        外部`compute_implicit_distance`通过`before_place`对坐标点进行了逆变换，只需要按局部坐标系计算即可
        """
        import pyvista as pv

        poly = pv.PolyData(mesh_cell_centers)
        surface = self.to_local_polydata().extract_surface()

        poly = poly.compute_implicit_distance(surface, inplace=True)

        if progress and self.progress_manually:
            progress.update(self.n_tasks)

        return poly['implicit_distance'].view(np.ndarray)

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
        from metalpy.utils.model import pv_ufunc_apply

        ret: pv.PolyData | pv.MultiBlock = self.to_local_polydata()

        if ret is not None:
            def transform(model):
                pts = self.transforms.transform(model.points)
                model.points = pts
            pv_ufunc_apply(ret, transform, inplace=True)

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

    def to_scene(self, model=True) -> 'Scene':
        """直接基于当前Shape3D实例构建场景

        Parameters
        ----------
        model
            模型值，默认为True，代表有效

        Returns
        -------
        scene
            仅包含当前Shape3D实例的场景Scene

        See Also
        --------
        :class:`metalpy.scab.modelling.Scene` : 用于建模的场景对象
        """
        from metalpy.scab.modelling import Scene
        return Scene.of(self, models=(model,), skip_checking=True)

    def build(self, model=True, **kwargs):
        """直接基于当前Shape3D实例构建场景的网格和模型

        Parameters
        ----------
        model
            模型值，默认为True，代表有效
        kwargs
            其它适用于Scene.build的参数

        Returns
        -------
        model_mesh
            模型网格

        See Also
        --------
            Scene.build : 构建场景的网格和模型
        """
        return self.to_scene(model=model).build(**kwargs)

    def plot(self, *args, **kwargs):
        """直接生成PyVista对象并调用plot在窗口中绘制

        Parameters
        ----------
        kwargs
            其它适用于pv.PolyData.plot的参数

        See Also
        --------
            Shape3D.to_polydata : 生成PyVista对象
        """
        self.to_polydata().plot(*args, **kwargs)

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

        ret = Bounds.unbounded(3)
        for i in range(ret.n_axes):
            val = bounds[:, i]
            val = val[~np.isnan(val)]

            if len(bounds) == 0:
                # 无上下界
                continue

            ret.set(i, min=val.min(), max=val.max())

        return Bounds(ret)

    @property
    def oriented_bounds(self) -> np.ndarray:
        """获取Shape在世界坐标系下的八点包围盒

        Returns
        -------
        ret : array(8, 3)
            Shape在世界坐标系下的八点包围盒[[x0, y0, z0], ...[x7, y7, z7]]

        Notes
        -----
        八点包围盒表示法下，无界会使用对应的正负无穷（inf）表示（Bounds约定nan为无界）

        各种空间变换由于精度问题，对无界 / 正负无穷（inf）边界的变换结果可能无法预测
        """
        bounds = self.local_oriented_bounds
        bounds = self.transforms.transform(bounds)

        return bounds

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

        默认实现基于to_local_polydata，存在性能问题，
        子类应当实现local_oriented_bounds和local_bounds中的一个

        Returns
        -------
        ret : array(6)
            Shape在局部坐标系下的长方体包围盒[x0, x1, y0, y1, z0, z1]
        """
        if type(self).local_oriented_bounds is Shape3D.local_oriented_bounds:
            return Bounds(self.to_local_polydata().bounds)
        else:
            bounds = self.local_oriented_bounds
            ret = np.zeros(6)
            ret[::2] = np.min(bounds, axis=1)
            ret[1::2] = np.max(bounds, axis=1)

            return Bounds(ret)

    @property
    def local_oriented_bounds(self) -> np.ndarray:
        """获取Shape在局部坐标系下的八点包围盒

        默认实现基于to_local_polydata，存在性能问题，
        子类应当实现local_oriented_bounds和local_bounds中的一个

        Returns
        -------
        ret : array(8, 3)
            Shape在局部坐标系下的八点包围盒[[x0, y0, z0], ...[x7, y7, z7]]

        Notes
        -----
        八点包围盒表示法下，无界会使用对应的正负无穷（inf）表示（Bounds约定nan为无界）
        """
        infs = [np.inf, -np.inf]

        local_bounds = self.local_bounds.to_inf_format()
        n_axes = local_bounds.n_axes

        xrng = local_bounds.xrange if n_axes > 0 else infs
        yrng = local_bounds.yrange if n_axes > 1 else infs
        zrng = local_bounds.zrange if n_axes > 2 else infs

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

    def _check_progress(self, progress):
        if progress is True and self.progress_manually:
            return tqdm.tqdm(total=self.n_tasks)
        else:
            return progress


def bounding_box_of(shapes: Iterable[Shape3D]):
    return reduce(union, (shape.bounds for shape in shapes))
