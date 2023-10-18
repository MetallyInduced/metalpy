import warnings
from functools import partial
from typing import Union

import numpy as np
import psutil
import taichi as ti
from discretize.base import BaseTensorMesh
from discretize.utils import mkvc

from metalpy.scab.utils.misc import Field
from metalpy.utils.string import format_string_list
from metalpy.utils.taichi import ti_cfg, ti_test_snode_support, ti_size_max
from .solvers.compressed import forward_compressed
from .solvers.integrated import forward_integrated
from .solvers.seperated import forward_seperated


class Demagnetization:
    Compressed = 'compressed'
    Seperated = 'seperated'
    Integrated = 'integrated'
    methods = {
        Compressed,
        Seperated,
        Integrated,
    }

    def __init__(
            self,
            mesh: BaseTensorMesh,
            source_field: Field,
            active_ind=None,
            method=None,
            compressed_size: Union[int, float, None] = None,
            deterministic: Union[bool, str] = True,
            verbose=False
    ):
        """
        通过CG或BiCGSTAB求解计算退磁作用下的磁化强度

        Parameters
        ----------
        mesh
            模型网格
        source_field
            外部场源
        active_ind
            有效网格下标或掩码
        method
            求解算法，参考`Demagnetization.methods`
        compressed_size
            压缩表尺寸，只针对`Demagnetization.Compressed`有效，用于指定压缩后的核矩阵哈希表最大大小，应尽可能小以满足内存要求。
            浮点数代表按完整核矩阵大小的比例设置，整数代表直接设置尺寸。
            如果大小不足或过大内部会给出提示信息
        deterministic
            采用确定性模式并行哈希，默认为True，确保核矩阵正确，但会牺牲一定计算效率和空间。
            False则会抛弃确定性约束，牺牲一定精度获取更优的时空间效率。
            指定为CompressedForward.Optimal时直接采用unique函数计算理想压缩结果
        verbose
            是否输出额外信息，对于某些求解器，可以输出一些辅助信息，例如输出taichi线性求解器的进度

        Notes
        -----
        compressed_size参考 `metalpy.scab.demag.solvers.compressed.forward_compressed` 定义

        三个方法具体为：
            `Integrated` 采用朴素算法，直接求解完整核矩阵

            `Seperated` 将核矩阵拆分为9份来一定程度上绕过taichi对矩阵大小的限制

            `Compressed` 通过哈希表将核矩阵压缩，以计算效率为代价换取大幅降低内存需求量

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
        self.verbose = verbose

        cell_centers = mesh.cell_centers
        if active_ind is not None:
            cell_centers = cell_centers[active_ind]

        self.receiver_locations = cell_centers

        # 计算网格在三个方向的边界位置
        bsw = cell_centers - mesh.h_gridded[active_ind] / 2.0
        tne = cell_centers + mesh.h_gridded[active_ind] / 2.0

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]
        zn1, zn2 = bsw[:, 2], tne[:, 2]

        self.Xn = np.c_[mkvc(xn1), mkvc(xn2)]
        self.Yn = np.c_[mkvc(yn1), mkvc(yn2)]
        self.Zn = np.c_[mkvc(zn1), mkvc(zn2)]

    def dpred(self, model):
        """计算退磁效应下的等效磁化率

        Parameters
        ----------
        model: array-like(nC,)
            磁化率模型

        Returns
        -------
        ret : array(nC, 3)
            三轴等效磁化率矩阵
        """
        nC = self.Xn.shape[0]
        H0 = self.source_field.unit_vector
        H0 = np.tile(H0[None, :], nC).ravel()
        X = np.tile(model, 3).ravel()
        b = X * H0

        base_cell_sizes = np.r_[
            self.mesh.h[0].min(),
            self.mesh.h[1].min(),
            self.mesh.h[2].min(),
        ]

        m = _predict(b, self.receiver_locations, self.Xn, self.Yn, self.Zn, base_cell_sizes, model,
                     method=self.method,
                     compressed_size=self.compressed_size,
                     deterministic=self.deterministic,
                     verbose=self.verbose
                     )
        return m.reshape(-1, 3)


def _predict(
        magnetization: np.ndarray,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        base_cell_sizes: np.ndarray,
        susc_model: np.ndarray,
        method=None,
        compressed_size: Union[int, float, None] = None,
        deterministic: Union[bool, str] = True,
        verbose=False
):
    is_cpu = ti_cfg().arch == ti.cpu

    mat_size = receiver_locations.shape[0] * 3 * xn.shape[0] * 3
    kernel_size = mat_size * np.finfo(np.result_type(receiver_locations, xn)).bits / 8

    if is_cpu:
        available_memory = psutil.virtual_memory().free
    else:
        # TODO: 看taichi什么时候能出一个查询剩余显存/内存的接口
        available_memory = None

    kw_int = {'is_cpu': is_cpu}
    kw_sep = {'is_cpu': is_cpu, 'verbose': verbose}
    kw_com = {'compressed_size': compressed_size, 'deterministic': deterministic, 'verbose': verbose}

    if method is not None:
        if method == Demagnetization.Compressed:
            candidate = partial(forward_compressed, **kw_com)
        elif method == Demagnetization.Seperated:
            candidate = partial(forward_seperated, direct_to_host=False, **kw_sep)
        elif method == Demagnetization.Integrated:
            if mat_size > ti_size_max:
                warnings.warn(f'`Integrated` method does not support'
                              f' kernel size ({mat_size}) > ti_size_max ({ti_size_max}),'
                              f' which may lead to unexpected result.')
            candidate = partial(forward_integrated, **kw_int)
        else:
            warnings.warn('Unrecognized solver. Falling back to `Seperated` method.')
            candidate = partial(forward_seperated, direct_to_host=False, **kw_sep)
    elif available_memory is not None and kernel_size > available_memory * 0.8:
        candidate = partial(forward_compressed, **kw_com)
    elif mat_size > ti_size_max:
        candidate = partial(forward_seperated, direct_to_host=False, **kw_sep)
    elif is_cpu:
        candidate = partial(forward_integrated, **kw_int)
    elif ti_test_snode_support():
        candidate = partial(forward_seperated, direct_to_host=True, **kw_sep)
    else:
        print("Current GPU doesn't support SNode, fall back to legacy implementation.")
        candidate = partial(forward_integrated, **kw_int)

    if is_cpu or candidate.func == forward_compressed:
        return candidate(magnetization, receiver_locations, xn, yn, zn, base_cell_sizes, susc_model)
    else:
        # CPU后端可以直接判断内存是否足够
        # 但CUDA后端不能在没有额外依赖的情况下查询内存，并且内存不足时会抛出异常，无法通过再次ti.init还原
        # 因此留一个错误信息以供参考
        try:
            return candidate(magnetization, receiver_locations, xn, yn, zn, base_cell_sizes, susc_model)
        except RuntimeError as _:
            raise RuntimeError(f'Failed to build kernel matrix.'
                               f' This may be resulted by insufficient memory.'
                               f' Try specify `method={Demagnetization.__name__}.Compressed`'
                               f' and adjust `compressed_size`'
                               f' to reduce memory consumption.')
