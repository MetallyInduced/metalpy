from __future__ import annotations

import enum
import warnings
from typing import TYPE_CHECKING, Sequence, Literal

import numpy as np
from discretize import TensorMesh
from numpy.typing import ArrayLike
from discretize.tree_mesh import TreeMesh, TreeCell

from metalpy.scab.modelling.modelled_mesh import ModelledMesh
from metalpy.utils.bounds import Bounds
from metalpy.utils.string import format_string_list

if TYPE_CHECKING:
    from typing import TypeVar
    from metalpy.scab.modelling import Scene

    TScene = TypeVar('TScene', bound=Scene)


class OctreeBuilder(enum.Enum):
    refine = enum.auto()
    simplify = enum.auto()


class TreeMeshBuilder:
    Octree = OctreeBuilder

    def build_tree(
            self: 'TScene', cell_size=None, n_cells=None, bounds=None,
            executor=None, progress=False,
            method: Literal['refine', 'simplify'] = Octree.refine,
            diagonal_balance=False,
            cache=None, cache_dir=None
    ) -> ModelledMesh:
        """根据给定的网格尺寸，构建场景的八叉树网格和模型

        Parameters
        ----------
        cell_size : number or array(3,)
            网格长宽高相等或分别定义网格x, y, z方向宽度大小
        n_cells : number or array(3,)
            总网格数或x, y, z方向网格数
        bounds : array(6,) or Bounds
            网格范围，为array(6,)[xmin, xmax, ymin, ymax, zmin, zmax]或Bounds实例，为None的位置会使用默认值
        executor
            并行执行器
        progress
            是否显示进度条
        method
            八叉树网格的构建方法，可选选项为 `'refine'` 或 `'simplify'` 。
            `'refine'` 构建效率更高，但是可能产生冗余的细分节点，
            `'simplify'` 可以构造最小八叉树，但是构建效率较低
        diagonal_balance
            是否需要进行对角线平衡化，使得对角线相邻的网格间细分层级差别不大于1
        cache
            指示缓存行为，详见Scene.build_model。单独使用cache=True来使用默认缓存路径或cache='...'来指定缓存文件路径
        cache_dir
            指示缓存行为，详见Scene.build_model。单独使用cache_dir='...'来指定缓存所在的文件夹路径

        Returns
        -------
        model_mesh
            八叉树模型网格

        Notes
        -----
        视模型复杂度和最小网格尺寸等因素：

        * `'refine'` 方法通过自顶向下的方法构建八叉树，因此其中会存在大量多余的顶级叶子节点，
          导致其有效网格数一般为 `'simplify'` 结果的 150% ~ 200%

        * `'simplify'` 方法通过自底向上的方法构建八叉树，运行效率较低，
          运行耗时一般为 `'refine'` 方法的 200% ~ 600% 或更高

        See Also
        --------
        Scene.create_mesh : 构造基础网格
        Scene.build_model : 构造模型
        TreeMeshBuilder.simplify : 从精细网格进行简化得到八叉树网格
        TreeMeshBuilder.refine_mesh : 从粗网格进行细分得到八叉树网格
        """
        if method == TreeMeshBuilder.Octree.refine:
            base_mesh = self.create_mesh(
                cell_size=cell_size,
                n_cells=n_cells,
                bounds=bounds
            )
            mesh = self.refine_mesh(
                base_mesh,
                ratio=1,
                diagonal_balance=diagonal_balance
            )
            ret = self.build_model(
                mesh,
                executor=executor,
                progress=progress
            )
        elif method == TreeMeshBuilder.Octree.simplify:
            base_model_mesh = self.build(
                cell_size=cell_size,
                n_cells=n_cells,
                bounds=bounds,
                executor=executor,
                progress=progress
            )
            ret = self.simplify(
                base_model_mesh,
                diagonal_balance=diagonal_balance
            )
        else:
            raise ValueError(
                f'Unknown octree builder `{repr(method)}` got,'
                f' expected {format_string_list(TreeMeshBuilder.Octree, multiline=True)}'
            )

        return ret

    @staticmethod
    def simplify(
            base_mesh: ModelledMesh | TensorMesh,
            scalars: str | ArrayLike | None = None,
            *,
            diagonal_balance=False
    ) -> ModelledMesh:
        """自底向上地对精细网格进行简化，生成树网格结构

        具体为以 2*2(*2) 的窗口对model进行判断，如果模型值相同，则合并为一个网格，
        否则不进行合并。通过逐级进行合并，构建树状结构

        Parameters
        ----------
        base_mesh
            需要进行简化的基础网格
        scalars
            需要进行合并的model或model名，
            指定向量时，在该model上进行合并判断，
            指定字符串时，从 `base_mesh` 中提取对应model，
            默认为None，选取 `base_mesh` 的默认model
        diagonal_balance
            是否需要进行对角线平衡化，使得对角线相邻的网格间细分层级差别不大于1

        Returns
        -------
        ret
            `base_mesh` 简化后得到的树网格，其中包含重映射后的 `model`

        Notes
        -----
        `discretize.TreeMesh` 中默认启用平衡化，保证直接相邻的网格间细分层级差别不大于1
        （2D下为4-连通近邻，3D下为6-连通近邻）

        当启用 `diagonal_balance=True` 时，会额外对对角线相邻的网格进行平衡化

        会自动检查并补充网格数为2的幂
        """
        if not isinstance(base_mesh, ModelledMesh):
            assert not isinstance(scalars, str), (
                f'Unexpected `scalars` got (scalars={scalars}).'
                f' A vector model must be provided to build'
                f' tree mesh on `base_mesh`.'
            )
            base_mesh = ModelledMesh(base_mesh, models=scalars)

        shape_cells = np.asarray(base_mesh.mesh.shape_cells)

        cells_required, expanded_cells = _check_shape_cells_power_of_2(shape_cells)
        # 将网格数扩展为2的幂数个
        if np.any(expanded_cells > 0):
            base_mesh = base_mesh.expand_cells(n_cells=expanded_cells)
            shape_cells = cells_required

        # 构建多级简化树
        base_model = base_mesh.get_complete_model(scalars).reshape(shape_cells, order='F')
        merge_masks, models = make_multilevel_mask(base_model)
        models = tuple(reversed(models))
        merge_masks = tuple(reversed(merge_masks))

        mesh = TreeMesh(base_mesh.mesh.h, origin=base_mesh.origin)

        def func(cell: TreeCell):
            merge_mask = merge_masks[cell._level]

            base = shape_cells // merge_mask.shape
            loc = cell._index_loc // base // 2

            return cell._level + ~merge_mask[tuple(loc)]

        mesh.refine(func, diagonal_balance=diagonal_balance)

        new_model = np.empty(len(mesh), dtype=base_model.dtype)
        for i, cell in enumerate(mesh):
            cell: TreeCell
            model = models[cell._level]
            base = shape_cells // model.shape
            loc = cell._index_loc // base // 2

            new_model[i] = model[tuple(loc)]

        if isinstance(scalars, str):
            models_data = {scalars: new_model}
        else:
            models_data = new_model

        return ModelledMesh(mesh, models=models_data)

    def refine_mesh(
            self: 'TScene',
            base_mesh: ModelledMesh | TensorMesh,
            ratio: int | Sequence[int] = 2,
            *,
            diagonal_balance=False
    ) -> TreeMesh:
        """自顶向下地对粗网格进行细分，生成树网格结构

        具体为将场景中的所有 `Shape3D` 实例导出为PyVista对象，获取其包含的所有三角面，
        通过判断每个三角面和网格的相交关系来进行网格细分

        Parameters
        ----------
        base_mesh
            需要进行细化的基础网格
        ratio
            网格细分比例，将原始网格中的每一格拆分为若干个子网格，支持通过数组分别指定三个轴的细分比例，例如 `[1, 2, 3]`
        diagonal_balance
            是否需要进行对角线平衡化，使得对角线相邻的网格间细分层级差别不大于1

        Returns
        -------
        ret
            `base_mesh` 细化后得到的树网格

        Notes
        -----
        `discretize.TreeMesh` 中默认启用平衡化，保证直接相邻的网格间细分层级差别不大于1
        （2D下为4-连通近邻，3D下为6-连通近邻）

        当启用 `diagonal_balance=True` 时，会额外对对角线相邻的网格进行平衡化

        `ratio` 参数允许分别指定三个方向的细分比例，
        例如 `[1, 2, 3]` 指定将每个网格在x、y、z方向上分割为1、2、3份，
        即将每个网格细分为(1 * 2 * 3=)6个子网格

        会在完成网格细分后检查并补充网格数为2的幂
        """
        if isinstance(base_mesh, ModelledMesh):
            base_mesh = base_mesh.mesh

        if np.ndim(ratio) < 1:
            ratio = [int(ratio)] * 3
        else:
            ratio = [int(r) for r in ratio]

        specs = list(base_mesh.h)
        refined_specs = [np.repeat(h / r, r) for h, r in zip(specs, ratio)]

        shape_cells = np.asarray([len(h) for h in refined_specs])

        # 将网格数扩展为2的幂数个
        cells_required, expanded_cells = _check_shape_cells_power_of_2(shape_cells)
        if np.any(expanded_cells > 0):
            for i, required in enumerate(expanded_cells.end):
                refined_specs[i] = np.r_[refined_specs[i], np.full(required, refined_specs[i][-1])]

        mesh = TreeMesh(refined_specs, origin=base_mesh.origin)

        models = self.to_multiblock()
        triangles = []
        for m in models:
            if len(m.faces) % 4 != 0 or np.any(m.faces[::4] != 3):
                # 判断模型是否已经三角化
                m.triangulate(inplace=True)
            triangles.append(m.points[m.faces.reshape(-1, 4)[:, 1:]])

        triangles = np.concatenate(triangles, axis=0)
        mesh.refine_triangle(triangles, levels=mesh.max_level, diagonal_balance=diagonal_balance)

        return mesh


def _check_shape_cells_power_of_2(shape_cells):
    # 检查各个方向上网格数是否为2的幂
    expanded_cells = np.zeros(2 * len(shape_cells), dtype=int).view(Bounds)
    cells_required = 1 << np.ceil(np.log2(shape_cells)).astype(int)
    for i, (n, required) in enumerate(zip(shape_cells, cells_required)):
        if n != required:
            warnings.warn(f' Got {n} cells on axis {i}, which is not power of 2.'
                          f' Will be expanded to {required} cells on this axis.')
            expanded_cells.set(i, max=required - n)

    return cells_required, expanded_cells


def reshape_fixed_window_fortran(a, windows_size, return_windows_axes=False):
    if np.size(windows_size) == 1:
        windows_size = np.full(np.ndim(a), windows_size)

    new_shape = []
    windows_axes = []
    for i, (size, windows_size) in enumerate(zip(a.shape, windows_size)):
        if windows_size is None:
            new_shape.append(size)
        else:
            if size % windows_size != 0:
                raise ValueError(f'Size must be an integer multiple of window size.'
                                 f' (size={size} and window_size={windows_size} on axis {i})')

            windows_axes.append(len(new_shape))

            new_shape.append(windows_size)
            new_shape.append(size // windows_size)

    reshaped = a.reshape(new_shape, order='F')

    if return_windows_axes:
        return reshaped, tuple(windows_axes)
    else:
        return reshaped


def make_multilevel_mask(model):
    """通过结构化正交网格的模型值，构建四叉树/八叉树层级结构

    以 2*2(*2) 的窗口对model进行判断，如果模型值相同，则合并为一个网格，否则不进行合并，逐级完成合并，构建树状结构

    Parameters
    ----------
    model
        结构化网格的模型值，维度与网格维度相同

    Returns
    -------
    models_and_merge_masks
        (models, merge_masks)，顺序为从自底向上，从最大细分等级到最小细分等级，其中：
        merge_mask表示各级下的每个网格是否由上一级网格合并得到，True则代表该网格细分的子网格可以合并，False代表不可合并需要维持细分
        models为各级下的每个网格的模型值（只有对应merge_mask为True时值才有意义）

    Notes
    -----
    SimPEG中标准的model一般为一维向量，可以通过如下方法转换为三维张量

    >>> model.reshape(mesh.shape_cells, order='F')

    在discretize中，TreeMesh允许各个方向网格数不一致（但仍强制各个方向网格数为2的幂）：

    * 如果三轴网格数相同，则初始级别为0，为单独一个网格。

    * 如果三轴网格数不同，则初始级别为1，包含所有多出的网格，从第2级开始三轴网格数相同。
      例如(4, 8, 16)，各轴超出的网格倍率为(1, 2, 4)，则第1级有(1 * 2 * 4 =)8个网格。
    """
    shape_cells = np.asarray(model.shape)
    n_min_cells = np.min(shape_cells)
    n_levels = int(np.log2(n_min_cells))

    is_uniform = np.all(shape_cells == shape_cells[0])

    merge_mask = np.ones_like(model, dtype=bool)
    models = [model]
    merge_masks = [merge_mask]

    for i in range(n_levels):
        model, windows_axes = reshape_fixed_window_fortran(model, windows_size=2, return_windows_axes=True)
        down_sampled = model[:1, :, :1, :, :1, :]

        merge_mask = np.all(reshape_fixed_window_fortran(merge_mask, windows_size=2), axis=windows_axes)
        merge_mask &= np.all(model == down_sampled, axis=windows_axes)

        model = down_sampled.squeeze(windows_axes)

        models.append(model)
        merge_masks.append(merge_mask)

    if not is_uniform:
        models.append(np.empty((0, 0, 0)))
        merge_masks.append(np.empty((0, 0, 0)))

    return merge_masks, models
