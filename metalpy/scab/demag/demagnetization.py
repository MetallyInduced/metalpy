from __future__ import annotations

import warnings

import numpy as np
import psutil
import taichi as ti
from discretize.base import BaseTensorMesh
from discretize.utils import mkvc

from metalpy.scab.utils.misc import Field
from metalpy.utils.string import format_string_list
from metalpy.utils.taichi import ti_cfg, ti_test_snode_support, ti_size_max
from .solvers.compressed import CompressedSolver
from .solvers.integrated import IntegratedSolver
from .solvers.seperated import SeperatedSolver
from .solvers.solver import DemagnetizationSolver


class Demagnetization:
    Compressed = 'Compressed'
    Seperated = 'Seperated'
    Integrated = 'Integrated'
    methods = {
        Compressed,
        Seperated,
        Integrated,
    }

    def __init__(
            self,
            mesh: BaseTensorMesh,
            source_field: Field | None = None,
            active_ind=None,
            method=None,
            compressed_size: int | float | None = None,
            deterministic: bool | str = True,
            quantized: bool | None = None,
            symmetric: bool | None = None,
            progress=False,
            kernel_dtype=None
    ):
        """
        通过共轭梯度法求解计算退磁作用下的三轴等效磁化率

        Parameters
        ----------
        mesh
            模型网格
        source_field
            外部场源，也可以在dpred时指定，主要为保持和Simulation的一致性
        active_ind
            有效网格下标或掩码
        method
            采用的数值求解算法，参考 `Demagnetization.methods` 。默认为None，自动选择。
            可选选项：
                - Demagnetization.Compressed
                - Demagnetization.Seperated
                - Demagnetization.Integrated
        compressed_size
            压缩表尺寸，只针对`Demagnetization.Compressed`有效，用于指定压缩后的核矩阵哈希表最大大小，应尽可能小以满足内存要求。
            浮点数代表按完整核矩阵大小的比例设置，整数代表直接设置尺寸。
            如果大小不足或过大内部会给出提示信息
        deterministic
            采用确定性模式并行哈希，默认为True，牺牲一定计算效率和空间确保核矩阵正确。
            指定为CompressedForward.Optimal时直接采用unique函数计算理想压缩结果
        quantized
            采用量化模式压缩核矩阵。
            True则启用压缩，使用更小位宽来存储核矩阵，减少一定的空间需求，略微牺牲计算效率。
            （可能会报要求尺寸为2的幂的性能警告）。
            在CPU模式下会对性能造成较大影响，因此默认不启用。
            在GPU模式下性能影响较小，因此默认启用
        symmetric
            指示是否启用对称模式（当网格为规则网格时可以启用）。
            True则在检测到网格符合条件时启用对称模式，以对称矩阵形式存储核矩阵，减少一定的空间需求。
            False则禁用对称模式，无论是否符合条件。
            在CPU模式下会对性能造成较大影响，因此默认不启用。
            在GPU模式下性能影响较小，因此默认启用
        progress
            是否输出求解进度条，默认为False不输出
        kernel_dtype
            核矩阵数据类型，默认为None，自动从输入数据推断

        Notes
        -----
        compressed_size参考 `metalpy.scab.demag.solvers.compressed.CompressedSolver` 定义

        三个方法具体为：

        - `Integrated` 采用朴素算法，直接求解完整核矩阵
        - `Seperated` 将核矩阵拆分为9份来一定程度上绕过taichi对矩阵大小的限制
        - `Compressed` 通过哈希表将核矩阵压缩，以计算效率为代价换取大幅降低内存需求量

        网格规模较小时 `Integrated` 计算效率最高，
        随着网格规模增加 `Integrated` 和 `Seperated` 计算耗时持平，
        但随着网格规模继续增加， `Integrated` 和 `Seperated` 的内存需求量相对于网格数平方级膨胀，
        如果超出内存限制，则需要采用 `Compressed` 来时间换空间。
        """
        super().__init__()
        assert method is None or method in Demagnetization.methods, \
            f'`method` must by one of {format_string_list(Demagnetization.methods)}, got `{method}`.'

        self.mesh = mesh
        self.source_field = source_field
        self.method = method
        self.compressed_size = compressed_size
        self.deterministic = deterministic
        self.quantized = quantized
        self.symmetric = symmetric
        self.progress = progress

        cell_centers = mesh.cell_centers
        if active_ind is not None:
            cell_centers = cell_centers[active_ind]

        self.receiver_locations = cell_centers

        # 计算网格在三个方向的边界位置
        h_gridded = mesh.h_gridded[active_ind]
        bsw = cell_centers - h_gridded / 2.0
        tne = cell_centers + h_gridded / 2.0

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]
        zn1, zn2 = bsw[:, 2], tne[:, 2]

        self.Xn = np.c_[mkvc(xn1), mkvc(xn2)]
        self.Yn = np.c_[mkvc(yn1), mkvc(yn2)]
        self.Zn = np.c_[mkvc(zn1), mkvc(zn2)]

        base_cell_sizes = np.r_[
            self.mesh.h[0].min(),
            self.mesh.h[1].min(),
            self.mesh.h[2].min(),
        ]

        self.is_cpu = ti_cfg().arch == ti.cpu

        # CPU后端可以直接判断内存是否足够
        # 但CUDA后端不能在没有额外依赖的情况下查询内存，并且内存不足时会抛出异常，无法通过再次ti.init还原
        # 因此留一个错误信息以供参考
        try:
            self.solver = dispatch_solver(
                self.receiver_locations, self.Xn, self.Yn, self.Zn, base_cell_sizes,
                source_field=self.source_field,
                is_cpu=self.is_cpu, method=self.method,
                compressed_size=self.compressed_size, deterministic=self.deterministic,
                quantized=self.quantized,
                symmetric=self.symmetric,
                progress=self.progress,
                kernel_dtype=kernel_dtype
            )
        except RuntimeError as e:
            if ti_cfg().arch == ti.cuda and 'cuStreamSynchronize' in e.args[0]:
                raise RuntimeError(f'Failed to build kernel matrix.'
                                   f' This may be resulted by insufficient memory'
                                   f' (please check log for details).'
                                   f'\nTo reduce memory consumption,'
                                   f' try specifying `kernel_dtype=np.float32`.'
                                   f'\nTo further improve memory efficiency,'
                                   f' use `method={Demagnetization.__name__}.Compressed`'
                                   f' with proper `compressed_size` and'
                                   f' enable `quantized=True` and `symmetric=True` when necessary.')
            else:
                raise e

    def dpred(self, model, source_field: Field | None = None):
        """计算退磁效应下的等效磁化率

        Parameters
        ----------
        model
            array-like(nC,)，磁化率模型
        source_field
            外部场源，覆盖求解器定义时给定的场源信息。
            如果为None，则采用给定的默认场源信息

        Returns
        -------
        ret
            array(nC, 3)，三轴等效磁化率矩阵
        """
        return self.solver.dpred(model, source_field=source_field).reshape(-1, 3)


def dispatch_solver(
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        base_cell_sizes: np.ndarray,
        source_field,
        is_cpu=True,
        method=None,
        compressed_size: int | float | None = None,
        deterministic: bool | str = True,
        quantized: bool = True,
        symmetric: bool = False,
        progress=False,
        kernel_dtype=None
) -> DemagnetizationSolver:
    n_obs = receiver_locations.shape[0]
    n_cells = xn.shape[0]
    mat_size = n_obs * 3 * n_cells * 3
    kernel_size = mat_size * np.finfo(np.result_type(receiver_locations, xn)).bits / 8

    max_cells_allowed_0 = int((1 + np.sqrt(1 + 8 * ti_size_max)) / 2)  # CompressedSolver对称模式下支持的最大网格数
    max_cells_allowed_1 = int(np.sqrt(ti_size_max))  # CompressedSolver非对称模式和SeperatedSolver支持的最大网格数
    max_cells_allowed_2 = max_cells_allowed_1 // 3  # IntegratedSolver支持的最大网格数

    if is_cpu:
        available_memory = psutil.virtual_memory().free
    else:
        # TODO: 看taichi什么时候能出一个查询剩余显存/内存的接口
        available_memory = None

    common_kwargs = {
        'receiver_locations': receiver_locations,
        'xn': xn,
        'yn': yn,
        'zn': zn,
        'base_cell_sizes': base_cell_sizes,
        'source_field': source_field,
        'kernel_dtype': kernel_dtype,
        'progress': progress,
    }
    kw_int = {**common_kwargs}
    kw_sep = {**common_kwargs}
    kw_com = {
        'compressed_size': compressed_size,
        'deterministic': deterministic,
        'quantized': quantized,
        'symmetric': symmetric,
        **common_kwargs
    }

    if n_cells > max_cells_allowed_0:
        raise AssertionError(
            f'Mesh with {n_cells} cells (> {max_cells_allowed_0})'
            f' is not supported.'
        )
    elif n_cells > max_cells_allowed_1:
        assert method in {Demagnetization.Compressed, None}, (
            f'Mesh with {n_cells} cells (> {max_cells_allowed_1})'
            f' requires symmetric `{CompressedSolver.__name__}` (got `{solver_name(method)}`).'
        )
        assert symmetric, (
            f'Mesh with {n_cells} cells (> {max_cells_allowed_1})'
            f' requires `symmetric` mesh.'
            f' Try setting `symmetric=True`.'
        )
    elif n_cells > max_cells_allowed_2:
        assert method != Demagnetization.Integrated, (
            f'`{IntegratedSolver.__name__}` does not support'
            f' mesh with {n_cells} cells (> {max_cells_allowed_2}).'
        )

    if method is not None:
        candidate = dispatch_solver_by_name(
            method,
            kw_int=kw_int,
            kw_sep=kw_sep,
            kw_com=kw_com
        )
    elif not is_cpu or available_memory is not None and kernel_size > available_memory * 0.8:
        candidate = CompressedSolver(**kw_com)
    elif n_cells > max_cells_allowed_1:
        candidate = CompressedSolver(**kw_com)
    elif n_cells > max_cells_allowed_2:
        candidate = SeperatedSolver(direct_to_host=ti_test_snode_support(), **kw_sep)
    else:
        candidate = IntegratedSolver(**kw_int)

    return candidate


def solver_name(method):
    return dispatch_solver_by_name(method).__name__


def dispatch_solver_by_name(method, kw_int=None, kw_sep=None, kw_com=None):
    """通过方法名获取求解器，当不指定求解器参数时，返回求解器类
    """
    if method == Demagnetization.Compressed:
        if kw_com is not None:
            candidate = CompressedSolver(**kw_com)
        else:
            candidate = CompressedSolver
    elif method == Demagnetization.Seperated:
        if kw_sep is not None:
            candidate = SeperatedSolver(direct_to_host=False, **kw_sep)
        else:
            candidate = SeperatedSolver
    elif method == Demagnetization.Integrated:
        if kw_int is not None:
            candidate = IntegratedSolver(**kw_int)
        else:
            candidate = IntegratedSolver
    else:
        warnings.warn('Unrecognized solver. Falling back to `Seperated` method.')
        if kw_sep is not None:
            candidate = SeperatedSolver(direct_to_host=False, **kw_sep)
        else:
            candidate = SeperatedSolver

    return candidate
