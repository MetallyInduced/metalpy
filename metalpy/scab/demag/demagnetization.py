from __future__ import annotations

import warnings

import numpy as np
import psutil
import taichi as ti
from discretize.base import BaseTensorMesh

from metalpy.scab.utils.misc import Field
from metalpy.utils.string import format_string_list
from metalpy.utils.taichi import ti_cfg, ti_test_snode_support, ti_size_max
from .solvers.compressed import CompressedSolver
from .solvers.demag_solver_context import DemagSolverContext
from .solvers.bce import BCESolver
from .solvers.indexed import IndexedSolver
from .solvers.integrated import IntegratedSolver
from .solvers.seperated import SeperatedSolver
from .solvers.solver import DemagnetizationSolver


class Demagnetization:
    BCE = 'BCE'
    Indexed = 'Indexed'
    Compressed = 'Compressed'
    Seperated = 'Seperated'
    Integrated = 'Integrated'
    methods = {
        BCE,
        Indexed,
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
            kernel_dtype=None,
            cutoff=np.inf
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
                - Demagnetization.BCE
                - Demagnetization.Indexed
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
        cutoff
            截断距离，当网格间距离超出截断距离，则不考虑之间的自退磁效应

        Notes
        -----
        `compressed_size` , `deterministic` , `quantized` , `symmetric` 参数
        参考 :class:`metalpy.scab.demag.solvers.compressed.CompressedSolver` 定义

        五个方法具体为：

        - `BCE` 采用BCE（Block Circulant Extension）算法，通过快速傅里叶变换加速核矩阵乘法，求解速度最快，是大规模规则网格下的最优求解器
        - `Compressed` 通过哈希表压缩核矩阵，大幅降低内存需求量，适用于任意网格，压缩率与网格规则度有关，规则网格上压缩率最高
        - `Indexed` 通过关系索引压缩核矩阵，结合截断距离使用实现线性空间复杂度，但求解速度较低，且只适用于大规模规则网格
        - `Seperated` 将核矩阵拆分为9份来一定程度上绕过taichi对矩阵大小的限制
        - `Integrated` 采用朴素算法，直接求解完整核矩阵

        如果采用规则网格， `BCE` 是最优选择；

        网格规模较小时 `Integrated` 计算效率最高；

        随着网格规模增加 `Integrated` 和 `Seperated` 计算耗时持平；

        但随着网格规模继续增加， `Integrated` 和 `Seperated` 的内存需求量相对于网格数平方级膨胀；

        如果超出内存限制，则需要采用 `Indexed` 或 `Compressed` 来时间换空间。
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

        self.is_cpu = ti_cfg().arch == ti.cpu

        context = DemagSolverContext(
            mesh, active_ind, source_field,
            kernel_dtype=kernel_dtype,
            cutoff=cutoff,
            progress=self.progress
        )

        # CPU后端可以直接判断内存是否足够
        # 但CUDA后端不能在没有额外依赖的情况下查询内存，并且内存不足时会抛出异常，无法通过再次ti.init还原
        # 因此留一个错误信息以供参考
        try:
            self.solver = dispatch_solver(
                context,
                is_cpu=self.is_cpu,
                method=self.method,
                compressed_size=self.compressed_size,
                deterministic=self.deterministic,
                quantized=self.quantized,
                symmetric=self.symmetric,
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
        return self.solver.dpred(model, source_field=source_field)


def dispatch_solver(
        context: DemagSolverContext,
        is_cpu=True,
        method=None,
        compressed_size: int | float | None = None,
        deterministic: bool | str = True,
        quantized: bool = True,
        symmetric: bool | None = None
) -> DemagnetizationSolver:
    n_obs = n_cells = context.n_active_cells
    mat_size = n_obs * 3 * n_cells * 3

    kernel_dtype = context.kernel_dtype
    if kernel_dtype is None:
        # 没有指定，临时猜测一个
        kernel_dtype = np.float64

    kernel_size = mat_size * np.finfo(kernel_dtype).bits / 8

    max_cells_allowed_0 = int((1 + np.sqrt(1 + 8 * ti_size_max)) / 2)  # CompressedSolver对称模式下支持的最大网格数
    max_cells_allowed_1 = int(np.sqrt(ti_size_max))  # CompressedSolver非对称模式和SeperatedSolver支持的最大网格数
    max_cells_allowed_2 = max_cells_allowed_1 // 3  # IntegratedSolver支持的最大网格数

    if is_cpu:
        available_memory = psutil.virtual_memory().free
    else:
        # TODO: 看taichi什么时候能出一个查询剩余显存/内存的接口
        available_memory = None

    support_any_cells = {Demagnetization.BCE, Demagnetization.Indexed, None}

    if n_cells > max_cells_allowed_0:
        available_solvers = support_any_cells
        choices = format_string_list(available_solvers - {None}, multiline=True, quote=solver_tag)

        assert method in available_solvers and context.is_active_cells_symmetric, (
            f'Mesh with {n_cells} cells (> {max_cells_allowed_0})'
            f' requires symmetric mesh with following solvers:'
            f' {choices}'
            f'\n(got `{solver_tag(method)}`).'
        )
    elif n_cells > max_cells_allowed_1:
        available_solvers = support_any_cells | {Demagnetization.Compressed}
        choices = format_string_list(available_solvers - {None}, multiline=True, quote=solver_tag)

        assert method in available_solvers, (
            f'Mesh with {n_cells} cells (> {max_cells_allowed_1})'
            f' requires following solvers:'
            f' {choices}'
            f'\n(got `{solver_tag(method)}`).'
        )

        if method == Demagnetization.Compressed:
            if symmetric is None:
                symmetric = CompressedSolver.Strict

            # 压缩求解器下必须在规则网格下启用对称模式
            assert symmetric, (
                f'Mesh with {n_cells} cells (> {max_cells_allowed_1})'
                f' requires `symmetric` mesh for {solver_tag(Demagnetization.Compressed)}.'
                f' Try setting `symmetric=True`.'
            )

        # TODO: 压缩求解器实现无索引矩阵模式后可以放宽该限制
        # 必须要规则网格才能支持65535个网格
        assert context.is_active_cells_symmetric, (
            f'Mesh with {n_cells} cells (> {max_cells_allowed_1})'
            f' requires `symmetric` mesh.'
        )
    elif n_cells > max_cells_allowed_2:
        assert method != Demagnetization.Integrated, (
            f'`{IntegratedSolver.__name__}` does not support'
            f' mesh with {n_cells} cells (> {max_cells_allowed_2}).'
        )

    common_kwargs = {
        'context': context,
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
    kw_ind = {**common_kwargs}
    kw_bce = {**common_kwargs}

    if method is not None:
        candidate = dispatch_solver_by_name(
            method,
            kw_int=kw_int,
            kw_sep=kw_sep,
            kw_com=kw_com,
            kw_ind=kw_ind,
            kw_bce=kw_bce,
        )
    elif n_cells > max_cells_allowed_0:
        if context.is_symmetric:
            # 规则网格优先采用BCE求解器
            candidate = BCESolver(**kw_bce)
        else:
            candidate = IndexedSolver(**kw_ind)
    elif (
            n_cells > max_cells_allowed_1  # 超出尺寸允许范围
            or not is_cpu  # 显卡默认需要进行压缩
            or available_memory is not None and kernel_size > available_memory * 0.8  # 超出设备内存限制
    ):
        if context.is_symmetric:
            # 规则网格优先采用BCE求解器
            candidate = BCESolver(**kw_bce)
        elif context.is_active_cells_symmetric:
            # 局部规则网格优先采用BCE求解器
            candidate = IndexedSolver(**kw_ind)
        else:
            candidate = CompressedSolver(**kw_com)
    elif n_cells > max_cells_allowed_2:
        candidate = SeperatedSolver(direct_to_host=ti_test_snode_support(), **kw_sep)
    else:
        candidate = IntegratedSolver(**kw_int)

    return candidate


def solver_name(method):
    return dispatch_solver_by_name(method).__name__


def solver_tag(method):
    return Demagnetization.__name__ + '.' + method


def dispatch_solver_by_name(method, kw_int=None, kw_sep=None, kw_com=None, kw_ind=None, kw_bce=None):
    """通过方法名获取求解器，当不指定求解器参数时，返回求解器类
    """
    if method == Demagnetization.BCE:
        if kw_bce is not None:
            candidate = BCESolver(**kw_bce)
        else:
            candidate = BCESolver
    elif method == Demagnetization.Indexed:
        if kw_ind is not None:
            candidate = IndexedSolver(**kw_ind)
        else:
            candidate = IndexedSolver
    elif method == Demagnetization.Compressed:
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
