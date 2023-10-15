import warnings
from functools import partial
from typing import Union

import numpy as np
import psutil
import pyamg
import taichi as ti
from discretize.base import BaseTensorMesh
from discretize.utils import mkvc
from taichi.lang.util import to_taichi_type

from metalpy.scab.utils.misc import Field
from metalpy.utils.string import format_string_list
from metalpy.utils.taichi import ti_kernel, ti_field, ti_FieldsBuilder, ti_cfg, ti_ndarray, ti_test_snode_support, \
    ti_func, copy_from, ti_size_max, ti_size_t


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
            compressed_size: Union[int, float, None] = None
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
        """
        super().__init__()
        assert method is None or method in Demagnetization.methods, \
            f'`method` must by one of {format_string_list(Demagnetization.methods)}, got `{method}`.'

        self.mesh = mesh
        self.source_field = source_field
        self.method = method
        self.compressed_size = compressed_size

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
                     method=self.method, compressed_size=self.compressed_size
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
        compressed_size: Union[int, float, None] = None
):
    mat_size = receiver_locations.shape[0] * 3 * xn.shape[0] * 3
    kernel_size = mat_size * np.finfo(np.result_type(receiver_locations, xn)).bits / 8

    is_cpu = ti_cfg().arch == ti.cpu

    if method is not None:
        if method == Demagnetization.Compressed:
            candidate = partial(forward_compressed, compressed_size=compressed_size)
        elif method == Demagnetization.Seperated:
            candidate = partial(forward_seperated, direct_to_host=False, is_cpu=is_cpu)
        else:
            candidate = partial(forward_integrated, is_cpu=is_cpu)
    elif is_cpu and kernel_size > psutil.virtual_memory().free * 0.8:
        # TODO: CUDA等后端是否有办法获知可用内存大小
        candidate = partial(forward_compressed, compressed_size=compressed_size)
    elif mat_size > ti_size_max:
        candidate = partial(forward_seperated, direct_to_host=False, is_cpu=is_cpu)
    elif is_cpu:
        candidate = partial(forward_integrated, is_cpu=is_cpu)
    elif ti_test_snode_support():
        candidate = partial(forward_seperated, direct_to_host=True, is_cpu=is_cpu)
    else:
        print("Current GPU doesn't support SNode, fall back to legacy implementation.")
        candidate = partial(forward_integrated, is_cpu=False)

    if is_cpu or candidate.func == forward_compressed:
        return candidate(magnetization, receiver_locations, xn, yn, zn, base_cell_sizes, susc_model)
    else:
        # CUDA后端内存不足时会抛出异常
        # CPU后端则则不会
        try:
            return candidate(magnetization, receiver_locations, xn, yn, zn, base_cell_sizes, susc_model)
        except RuntimeError as _:
            candidate = partial(forward_compressed, compressed_size=compressed_size)
            return candidate(magnetization, receiver_locations, xn, yn, zn, base_cell_sizes, susc_model)


def forward_integrated(
        magnetization: np.ndarray,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        base_cell_sizes: np.ndarray,
        susc_model: np.ndarray,
        is_cpu: bool
):
    """该函数直接计算核矩阵

    Parameters
    ----------
    magnetization
        磁化强度
    receiver_locations
        观测点
    xn, yn, zn
        网格边界
    base_cell_sizes
        网格最小单元大小
    susc_model
        磁化率模型
    is_cpu
        是否是CPU架构，否则需要在对应设备上分配返回值矩阵

    Notes
    -----
        优势：
        1. 简单直观

        缺陷：
        1. 存在taichi的int32索引限制
    """
    nC = xn.shape[0]
    nObs = receiver_locations.shape[0]

    if is_cpu:
        A = np.empty((3 * nObs, 3 * nC), dtype=np.float64)
    else:
        A = ti_ndarray(dtype=ti.f64, shape=(3 * nObs, 3 * nC))

    kernel_matrix_forward(receiver_locations, xn, yn, zn, base_cell_sizes, susc_model,
                          *[0] * 9, A, write_to_mat=True, compressed=False)

    if is_cpu:
        Amat = A
    else:
        Amat = A.to_numpy()

    return solve_Ax_b(Amat, magnetization)


def search_prime(value):
    """搜索合适的质数，默认从good primes中取，备选选项为更大的一个质数

    Parameters
    ----------
    value
        需要的质数的上界

    Returns
    -------
    prime
        小于value的某个合适的质数

    References
    ----------
    寻找哈希表的好质数： https://planetmath.org/goodhashtableprimes
    寻找下一个质数： https://www.numberempire.com/primenumbers.php
    """
    primes = {
        1 << 0: [1, 1],  # 别问
        1 << 1: [1, 2],
        1 << 2: [2, 3],
        1 << 3: [5, 7],
        1 << 4: [11, 13],
        1 << 5: [23, 29],
        1 << 6: [53, 59],
        1 << 7: [97, 101],
        1 << 8: [193, 197],
        1 << 9: [389, 397],
        1 << 10: [769, 773],
        1 << 11: [1543, 1549],
        1 << 12: [3079, 3083],
        1 << 13: [6151, 6163],
        1 << 14: [12289, 12301],
        1 << 15: [24593, 24611],
        1 << 16: [49157, 49169],
        1 << 17: [98317, 98321],
        1 << 18: [196613, 196643],
        1 << 19: [393241, 393247],
        1 << 20: [786433, 786449],
        1 << 21: [1572869, 1572871],
        1 << 22: [3145739, 3145741],
        1 << 23: [6291469, 6291487],
        1 << 24: [12582917, 12582919],
        1 << 25: [25165843, 25165853],
        1 << 26: [50331653, 50331683],
        1 << 27: [100663319, 100663327],
        1 << 28: [201326611, 201326621],
        1 << 29: [402653189, 402653201],
        1 << 30: [805306457, 805306459],
        1 << 31: [1610612741, 1610612747],
    }

    key = 1 << int(np.log2(value))
    candidates = primes[key]
    if candidates[0] == value:
        return candidates[1]
    else:
        return candidates[0]


def compress_kernel(
        Tmat33,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        indices_mat: np.ndarray
):
    # TODO: 等taichi更新获取变量类型的泛型操作后可以简化
    val_type = np.result_type(receiver_locations, xn)
    assert np.issubdtype(val_type, np.floating)
    val_type = getattr(np, str(val_type).replace('float', 'uint'))
    precision = np.iinfo(val_type).bits

    overflow = ti_field(ti.i8, 1)
    table_size = Tmat33[0][0].shape[0]
    kernel_size = np.prod(indices_mat.shape)
    prime = search_prime(table_size)

    hash_unique(
        receiver_locations,
        xn, yn, zn,
        *Tmat33[0], *Tmat33[1],
        indices_mat,
        to_taichi_type(val_type),
        precision,
        overflow,
        prime
    )

    overflow = overflow.to_numpy()[0]
    if overflow:
        raise RuntimeError(f'Compression hash table overflowed.'
                           f' Consider using a larger `compressed_size` (currently {table_size}).')

    used = np.count_nonzero(~np.isnan(Tmat33[0][0].to_numpy()))
    memory_efficiency = used / table_size
    if memory_efficiency < 0.95:
        # 一个特点是，尤其是对于规则六边形网格
        # 随着网格数增加，不同网格关系数的增速比核矩阵的增速慢
        # 即随着网格数增加，压缩比会越来越低
        warnings.warn(f'Low memory efficiency detected ({memory_efficiency:.2%}),'
                      f' consider set `compressed_size` to'
                      f' `{int(used / 0.95)}`'
                      f' or'
                      f' `1 / {int(1 / (used / 0.95 / kernel_size))}`'
                      f'.')


def forward_compressed(
        magnetization: np.ndarray,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        base_cell_sizes: np.ndarray,
        susc_model: np.ndarray,
        compressed_size: Union[int, float, None],
):
    """该函数将矩阵分为9个部分，计算后再拼接，从而实现一些优化

    Parameters
    ----------
    magnetization
        磁化强度
    receiver_locations
        观测点
    xn, yn, zn
        网格边界
    base_cell_sizes
        网格最小单元大小
    susc_model
        磁化率模型
    compressed_size
        压缩表尺寸，
        如果为整数，则用于指定尺寸大小，
        如果为浮点数则设置为核矩阵大小成比例设置，
        默认None则为1 / 32

    Notes
    -----
        优势：
        1. 可以绕过最大索引为的int32限制
        2. 可以使用AoS提高内存近邻性

        缺陷：
        1. 如果是在CPU架构上允许，由于会同时分配返回值矩阵和计算用的9个矩阵组，导致内存占用相比forward翻倍
        2. 如果启用direct_to_host会导致在一些旧GPU上无法使用，并且direct_to_host仍然无法绕过taichi的int32索引限制
        3. 如果不启用direct_to_host，则可以规避索引限制，但是会导致额外 sizeof(mat) / 9 的内存overhead
    """
    with ti_FieldsBuilder() as builder:
        nC = xn.shape[0]
        nObs = receiver_locations.shape[0]

        if compressed_size is None:
            compressed_size = 1 / 32

        if np.issubdtype(type(compressed_size), np.floating):
            compressed_size = int(nC * nObs * compressed_size)

        # [[Txx, Txy, Txz], [Tyx, Tyy, Tyz], [Tzx, Tzy, Tzz]]
        Tmat33 = [
            [ti_field(ti.f64) for _ in range(3)]
            for _ in range(3)
        ]
        Tmat9 = [t for ts in Tmat33 for t in ts]
        indices_mat = ti_field(ti_size_t)  # TODO: taichi当前的索引类型

        # 保证 Tx[xyz]，Ty[xyz]，Tz[xyz] 分别在空间上连续，提高空间近邻性
        for m in Tmat33:
            builder.dense(ti.i, compressed_size).place(*m)
        builder.dense(ti.ij, (nObs, nC)).place(indices_mat)
        builder.finalize()

        compress_kernel(Tmat33, receiver_locations, xn, yn, zn, indices_mat)

        kernel_matrix_forward(receiver_locations, xn, yn, zn, base_cell_sizes, susc_model,
                              *Tmat9, np.empty(0), write_to_mat=False, compressed=True)

        return solve_Tx_b_compressed(Tmat33, indices_mat, magnetization)


def forward_seperated(
        magnetization: np.ndarray,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        base_cell_sizes: np.ndarray,
        susc_model: np.ndarray,
        direct_to_host: bool,
        is_cpu: bool
):
    """该函数将矩阵分为9个部分，计算后再拼接，从而实现一些优化

    Parameters
    ----------
    magnetization
        磁化强度
    receiver_locations
        观测点
    xn, yn, zn
        网格边界
    base_cell_sizes
        网格最小单元大小
    susc_model
        磁化率模型
    direct_to_host
        是否通过kernel将结果直接zero-copy地复制到待返回的numpy数组
    is_cpu
        是否是CPU架构，如果是，才会在内存可用时合并核矩阵然后求解

    Notes
    -----
        优势：
        1. 可以绕过最大索引为的int32限制
        2. 可以使用AoS提高内存近邻性

        缺陷：
        1. 如果是在CPU架构上允许，由于会同时分配返回值矩阵和计算用的9个矩阵组，导致内存占用相比forward翻倍
        2. 如果启用direct_to_host会导致在一些旧GPU上无法使用，并且direct_to_host仍然无法绕过taichi的int32索引限制
        3. 如果不启用direct_to_host，则可以规避索引限制，但是会导致额外 sizeof(mat) / 9 的内存overhead
    """
    with ti_FieldsBuilder() as builder:
        nC = xn.shape[0]
        nObs = receiver_locations.shape[0]

        # [[Txx, Txy, Txz], [Tyx, Tyy, Tyz], [Tzx, Tzy, Tzz]]
        Tmat33 = [
            [ti_field(ti.f64) for _ in range(3)]
            for _ in range(3)
        ]
        Tmat9 = [t for ts in Tmat33 for t in ts]

        # 保证 Tx[xyz]，Ty[xyz]，Tz[xyz] 分别在空间上连续，提高空间近邻性
        for m in Tmat33:
            builder.dense(ti.ij, (nObs, nC)).place(*m)

        builder.finalize()

        kernel_matrix_forward(receiver_locations, xn, yn, zn, base_cell_sizes, susc_model,
                              *Tmat9, np.empty(0), write_to_mat=False, compressed=False)

        if is_cpu and psutil.virtual_memory().percent < 45:
            # TODO: 进一步考虑模型大小来选择求解方案，模型较大时也应该使用 `solve_Tx_b`
            return solve_Ax_b(merge_Tmat_as_A(Tmat33, direct_to_host=direct_to_host), magnetization)
        else:
            return solve_Tx_b(Tmat33, magnetization)


@ti_kernel
def hash_unique(
        receiver_locations: ti.types.ndarray(),
        xn: ti.types.ndarray(),
        yn: ti.types.ndarray(),
        zn: ti.types.ndarray(),
        Txx: ti.template(),
        Txy: ti.template(),
        Txz: ti.template(),
        Tyx: ti.template(),
        Tyy: ti.template(),
        Tyz: ti.template(),
        indices_mat: ti.template(),
        hash_type: ti.template(),
        precision: ti.template(),
        overflow: ti.template(),
        prime: ti.template()
):
    for I in ti.grouped(Txx):
        Txx[I] = ti.math.nan

    nC = xn.shape[0]
    nObs = receiver_locations.shape[0]
    table_size = Txx.shape[0]
    overflow[0] = 0
    hhp = precision // 4

    for iobs, icell in ti.ndrange(nObs, nC):
        if overflow[0] > 0:
            continue

        dz2 = zn[icell, 1] - receiver_locations[iobs, 2]
        dz1 = zn[icell, 0] - receiver_locations[iobs, 2]
        dy2 = yn[icell, 1] - receiver_locations[iobs, 1]
        dy1 = yn[icell, 0] - receiver_locations[iobs, 1]
        dx2 = xn[icell, 1] - receiver_locations[iobs, 0]
        dx1 = xn[icell, 0] - receiver_locations[iobs, 0]

        p0 = ti.bit_cast(dx1, hash_type)
        p1 = ti.bit_cast(dx2, hash_type)
        p2 = ti.bit_cast(dy1, hash_type)
        p3 = ti.bit_cast(dy2, hash_type)
        p4 = ti.bit_cast(dz1, hash_type)
        p5 = ti.bit_cast(dz2, hash_type)

        # 哈希函数设计原则：
        # 由于p[i]是直接由d[xyz][12]从二进制角度转换过来
        # 因此其中指数部分（高位）应在大部分情况下相同，导致哈希空间稀疏，
        # 所以需要采用循环位移来使其充分混淆
        a0: ti_size_t = (
            cshift(p1, hhp * 0)
            ^ cshift(p4, hhp * 1)
            ^ cshift(p2, hhp * 2)
            ^ cshift(p5, hhp * 3)
            ^ cshift(p3, hhp * 0)
            ^ cshift(p0, hhp * 1)
        ) % table_size

        a = a0
        while not ti.math.isnan(Txx[a]):
            if (
                Txx[a] == dx1
                and Txy[a] == dx2
                and Txz[a] == dy1
                and Tyx[a] == dy2
                and Tyy[a] == dz1
                and Tyz[a] == dz2
            ):
                break

            # 采用与nC/nObs互质的距离来做跳跃rehash：
            # 降低二次冲突概率，同时保证能遍历全表
            a += prime
            if a >= table_size:
                a -= table_size

            if a == a0:
                print('寄！')
                overflow[0] = 1
                break

        # 容量超了的话，其实已经不需要管下面数据错误的问题了
        # if capacity[0] <= 0:
        #     # print('寄！')
        #     continue

        indices_mat[iobs, icell] = a

        if ti.math.isnan(Txx[a]):
            Txx[a] = dx1
            Txy[a] = dx2
            Txz[a] = dy1
            Tyx[a] = dy2
            Tyy[a] = dz1
            Tyz[a] = dz2


@ti_func
def cshift(key, b):
    return (key << b) + (key >> b)


def solve_Ax_b(A, m):
    x, _ = pyamg.krylov.bicgstab(A, m)
    return x


def solve_Tx_b(Tmat33, m):
    with ti_FieldsBuilder() as builder:
        x = builder.place_dense_like(m[:, None])
        b = builder.place_dense_like(m[:, None])

        builder.finalize()

        copy_from(b, m[:, None])

        @ti_func
        def mul3(Ax, x, i: ti.template()):
            multiply_with_row_stride_3(
                Ax, *Tmat33[i], x, i, 3
            )

        @ti_kernel
        def linear(x: ti.template(), Ax: ti.template()):
            mul3(Ax, x, 0)
            mul3(Ax, x, 1)
            mul3(Ax, x, 2)

        ti.linalg.taichi_cg_solver(ti.linalg.LinearOperator(linear), b, x)

        return x.to_numpy()



def solve_Tx_b_compressed(Tmat33, indices_mat, m):
    with ti_FieldsBuilder() as builder:
        x = builder.place_dense_like(m[:, None])
        b = builder.place_dense_like(m[:, None])

        builder.finalize()

        copy_from(b, m[:, None])

        @ti_func
        def mul3(Ax, x, i: ti.template()):
            multiply_with_row_stride_3_compressed(
                Ax, indices_mat, *Tmat33[i], x, i, 3
            )

        @ti_kernel
        def linear(x: ti.template(), Ax: ti.template()):
            mul3(Ax, x, 0)
            mul3(Ax, x, 1)
            mul3(Ax, x, 2)

        ti.linalg.taichi_cg_solver(ti.linalg.LinearOperator(linear), b, x, quiet=False)

        return x.to_numpy()


def merge_Tmat_as_A(Tmat33, direct_to_host=True):
    nObs, nC = Tmat33[0][0].shape
    Amat = np.empty((3 * nObs, 3 * nC), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if direct_to_host:
                tensor_to_ext_arr(Tmat33[i][j], Amat, i, 3, j, 3)
            else:
                Amat[i::3, j::3] = Tmat33[i][j].to_numpy()

    return Amat


@ti_kernel
def kernel_matrix_forward(
        receiver_locations: ti.types.ndarray(),
        xn: ti.types.ndarray(),
        yn: ti.types.ndarray(),
        zn: ti.types.ndarray(),
        base_cell_sizes: ti.types.ndarray(),
        susc_model: ti.types.ndarray(),
        Txx: ti.template(),
        Txy: ti.template(),
        Txz: ti.template(),
        Tyx: ti.template(),
        Tyy: ti.template(),
        Tyz: ti.template(),
        Tzx: ti.template(),
        Tzy: ti.template(),
        Tzz: ti.template(),
        mat: ti.types.ndarray(),
        write_to_mat: ti.template(),
        compressed: ti.template(),
):
    # calculates A = I - X @ T, where T is the forward kernel, s.t. T @ m_v = B_v
    # m_v and B_v are both channel first (Array of Structure in taichi)
    # m_v = [Mx1, My1, Mz1, ... Mxn, Myn, Mzn]
    # B_v = [Bx1, By1, Bz1, ... Bxn, Byn, Bzn]
    #     ┌─────────────────────────── nC ─────────────────────────┐
    #     ┌────────────────┬────────────────┬─────┬────────────────┐ ─┐
    #     │ Txx, Txy, Txz, │ Txx, Txy, Txz, │ ... │ Txx, Txy, Txz, │  │
    #     │ Tyx, Tyy, Tyz, │ Tyx, Tyy, Tyz, │ ... │ Tyx, Tyy, Tyz, │  │
    #     │ Tzx, Tzy, Tzz, │ Tzx, Tzy, Tzz, │ ... │ Tzx, Tzy, Tzz, │  │
    #     ├────────────────┼────────────────┼─────┼────────────────┤  │
    # T = │ ...            │ ...            │ ... │ ...            │ nObs
    #     ├────────────────┼────────────────┼─────┼────────────────┤  │
    #     │ Txx, Txy, Txz, │ Txx, Txy, Txz, │ ... │ Txx, Txy, Txz, │  │
    #     │ Tyx, Tyy, Tyz, │ Tyx, Tyy, Tyz, │ ... │ Tyx, Tyy, Tyz, │  │
    #     │ Tzx, Tzy, Tzz, │ Tzx, Tzy, Tzz, │ ... │ Tzx, Tzy, Tzz, │  │
    #     └────────────────┴────────────────┴─────┴────────────────┘ ─┘
    tol1 = 1e-10  # Tolerance 1 for numerical stability over nodes and edges
    tol2 = 1e-4  # Tolerance 2 for numerical stability over nodes and edges

    # number of cells in mesh
    nC = xn.shape[0]
    nObs = receiver_locations.shape[0]

    # base cell dimensions
    min_hx = base_cell_sizes[0]
    min_hy = base_cell_sizes[1]
    min_hz = base_cell_sizes[2]

    dummy = xn[0, 0]
    total = nObs * nC
    if ti.static(compressed):
        total = Txx.shape[0]

    for i in range(total):
        iobs, icell = 0, 0
        if ti.static(not compressed):
            iobs, icell = i // nC, i % nC

        # comp. pos. differences for tne, bsw nodes. Adjust if location within
        # tolerance of a node or edge
        dx1, dx2, dy1, dy2, dz1, dz2 = dummy, dummy, dummy, dummy, dummy, dummy
        ti_use(dx1)
        ti_use(dx2)
        ti_use(dy1)
        ti_use(dy2)
        ti_use(dz1)
        ti_use(dz2)

        if ti.static(not compressed):
            dx1 = xn[icell, 0] - receiver_locations[iobs, 0]
            dx2 = xn[icell, 1] - receiver_locations[iobs, 0]
            dy1 = yn[icell, 0] - receiver_locations[iobs, 1]
            dy2 = yn[icell, 1] - receiver_locations[iobs, 1]
            dz1 = zn[icell, 0] - receiver_locations[iobs, 2]
            dz2 = zn[icell, 1] - receiver_locations[iobs, 2]
        else:
            dx1 = Txx[i]

            if ti.math.isnan(dx1):
                continue

            dx2 = Txy[i]
            dy1 = Txz[i]
            dy2 = Tyx[i]
            dz1 = Tyy[i]
            dz2 = Tyz[i]

        if ti.abs(dx1) / min_hx < tol2:
            dx1 = tol2 * min_hx
        if ti.abs(dx2) / min_hx < tol2:
            dx2 = tol2 * min_hx

        if ti.abs(dy1) / min_hy < tol2:
            dy1 = tol2 * min_hy
        if ti.abs(dy2) / min_hy < tol2:
            dy2 = tol2 * min_hy

        if ti.abs(dz1) / min_hz < tol2:
            dz1 = tol2 * min_hz
        if ti.abs(dz2) / min_hz < tol2:
            dz2 = tol2 * min_hz

        # comp. squared diff
        dx2dx2 = dx2 ** 2.0
        dx1dx1 = dx1 ** 2.0

        dy2dy2 = dy2 ** 2.0
        dy1dy1 = dy1 ** 2.0

        dz2dz2 = dz2 ** 2.0
        dz1dz1 = dz1 ** 2.0

        # 2D radius component squared of corner nodes
        R1 = dy2dy2 + dx2dx2
        R2 = dy2dy2 + dx1dx1
        R3 = dy1dy1 + dx2dx2
        R4 = dy1dy1 + dx1dx1

        # radius to each cell node
        r1 = ti.sqrt(dz2dz2 + R2)
        r2 = ti.sqrt(dz2dz2 + R1)
        r3 = ti.sqrt(dz1dz1 + R1)
        r4 = ti.sqrt(dz1dz1 + R2)
        r5 = ti.sqrt(dz2dz2 + R3)
        r6 = ti.sqrt(dz2dz2 + R4)
        r7 = ti.sqrt(dz1dz1 + R4)
        r8 = ti.sqrt(dz1dz1 + R3)

        # compactify argument calculations
        arg1_ = dx1 + dy2 + r1
        arg1 = dy2 + dz2 + r1
        arg2 = dx1 + dz2 + r1
        arg3 = dx1 + r1
        arg4 = dy2 + r1
        arg5 = dz2 + r1

        arg6_ = dx2 + dy2 + r2
        arg6 = dy2 + dz2 + r2
        arg7 = dx2 + dz2 + r2
        arg8 = dx2 + r2
        arg9 = dy2 + r2
        arg10 = dz2 + r2

        arg11_ = dx2 + dy2 + r3
        arg11 = dy2 + dz1 + r3
        arg12 = dx2 + dz1 + r3
        arg13 = dx2 + r3
        arg14 = dy2 + r3
        arg15 = dz1 + r3

        arg16_ = dx1 + dy2 + r4
        arg16 = dy2 + dz1 + r4
        arg17 = dx1 + dz1 + r4
        arg18 = dx1 + r4
        arg19 = dy2 + r4
        arg20 = dz1 + r4

        arg21_ = dx2 + dy1 + r5
        arg21 = dy1 + dz2 + r5
        arg22 = dx2 + dz2 + r5
        arg23 = dx2 + r5
        arg24 = dy1 + r5
        arg25 = dz2 + r5

        arg26_ = dx1 + dy1 + r6
        arg26 = dy1 + dz2 + r6
        arg27 = dx1 + dz2 + r6
        arg28 = dx1 + r6
        arg29 = dy1 + r6
        arg30 = dz2 + r6

        arg31_ = dx1 + dy1 + r7
        arg31 = dy1 + dz1 + r7
        arg32 = dx1 + dz1 + r7
        arg33 = dx1 + r7
        arg34 = dy1 + r7
        arg35 = dz1 + r7

        arg36_ = dx2 + dy1 + r8
        arg36 = dy1 + dz1 + r8
        arg37 = dx2 + dz1 + r8
        arg38 = dx2 + r8
        arg39 = dy1 + r8
        arg40 = dz1 + r8

        txx = (
            -2 * ti.atan2(dx1, arg1 + tol1)
            - -2 * ti.atan2(dx2, arg6 + tol1)
            + -2 * ti.atan2(dx2, arg11 + tol1)
            - -2 * ti.atan2(dx1, arg16 + tol1)
            + -2 * ti.atan2(dx2, arg21 + tol1)
            - -2 * ti.atan2(dx1, arg26 + tol1)
            + -2 * ti.atan2(dx1, arg31 + tol1)
            - -2 * ti.atan2(dx2, arg36 + tol1)
        ) / -4 / ti.math.pi

        txy = (
            ti.log(arg5)
            - ti.log(arg10)
            + ti.log(arg15)
            - ti.log(arg20)
            + ti.log(arg25)
            - ti.log(arg30)
            + ti.log(arg35)
            - ti.log(arg40)
        ) / -4 / ti.math.pi

        txz = (
            ti.log(arg4) - ti.log(arg9)
            + ti.log(arg14) - ti.log(arg19)
            + ti.log(arg24) - ti.log(arg29)
            + ti.log(arg34) - ti.log(arg39)
        ) / -4 / ti.math.pi

        tyx = (
            ti.log(arg5)
            - ti.log(arg10)
            + ti.log(arg15)
            - ti.log(arg20)
            + ti.log(arg25)
            - ti.log(arg30)
            + ti.log(arg35)
            - ti.log(arg40)
        ) / -4 / ti.math.pi

        tyy = (
            -2 * ti.atan2(dy2, arg2 + tol1)
            - -2 * ti.atan2(dy2, arg7 + tol1)
            + -2 * ti.atan2(dy2, arg12 + tol1)
            - -2 * ti.atan2(dy2, arg17 + tol1)
            + -2 * ti.atan2(dy1, arg22 + tol1)
            - -2 * ti.atan2(dy1, arg27 + tol1)
            + -2 * ti.atan2(dy1, arg32 + tol1)
            - -2 * ti.atan2(dy1, arg37 + tol1)
        ) / -4 / ti.math.pi

        tyz = (
            ti.log(arg3) - ti.log(arg8)
            + ti.log(arg13) - ti.log(arg18)
            + ti.log(arg23) - ti.log(arg28)
            + ti.log(arg33) - ti.log(arg38)
        ) / -4 / ti.math.pi

        tzx = (
            ti.log(arg4)
            - ti.log(arg9)
            + ti.log(arg14)
            - ti.log(arg19)
            + ti.log(arg24)
            - ti.log(arg29)
            + ti.log(arg34)
            - ti.log(arg39)
        ) / -4 / ti.math.pi

        tzy = (
            ti.log(arg3) - ti.log(arg8)
            + ti.log(arg13) - ti.log(arg18)
            + ti.log(arg23) - ti.log(arg28)
            + ti.log(arg33) - ti.log(arg38)
        ) / -4 / ti.math.pi

        tzz = (
            -2 * ti.atan2(dz2, arg1_ + tol1)
            - -2 * ti.atan2(dz2, arg6_ + tol1)
            + -2 * ti.atan2(dz1, arg11_ + tol1)
            - -2 * ti.atan2(dz1, arg16_ + tol1)
            + -2 * ti.atan2(dz2, arg21_ + tol1)
            - -2 * ti.atan2(dz2, arg26_ + tol1)
            + -2 * ti.atan2(dz1, arg31_ + tol1)
            - -2 * ti.atan2(dz1, arg36_ + tol1)
        ) / -4 / ti.math.pi

        neg_sus = -susc_model[icell]

        if ti.static(compressed):
            Txx[i] = neg_sus * txx
            Txy[i] = neg_sus * txy
            Txz[i] = neg_sus * txz
            Tyx[i] = neg_sus * tyx
            Tyy[i] = neg_sus * tyy
            Tyz[i] = neg_sus * tyz
            Tzx[i] = neg_sus * tzx
            Tzy[i] = neg_sus * tzy
            Tzz[i] = neg_sus * tzz
        else:
            if ti.static(write_to_mat):
                mat[iobs * 3 + 0, icell * 3 + 0] = neg_sus * txx
                mat[iobs * 3 + 0, icell * 3 + 1] = neg_sus * txy
                mat[iobs * 3 + 0, icell * 3 + 2] = neg_sus * txz
                mat[iobs * 3 + 1, icell * 3 + 0] = neg_sus * tyx
                mat[iobs * 3 + 1, icell * 3 + 1] = neg_sus * tyy
                mat[iobs * 3 + 1, icell * 3 + 2] = neg_sus * tyz
                mat[iobs * 3 + 2, icell * 3 + 0] = neg_sus * tzx
                mat[iobs * 3 + 2, icell * 3 + 1] = neg_sus * tzy
                mat[iobs * 3 + 2, icell * 3 + 2] = neg_sus * tzz
            else:
                Txx[iobs, icell] = neg_sus * txx
                Txy[iobs, icell] = neg_sus * txy
                Txz[iobs, icell] = neg_sus * txz
                Tyx[iobs, icell] = neg_sus * tyx
                Tyy[iobs, icell] = neg_sus * tyy
                Tyz[iobs, icell] = neg_sus * tyz
                Tzx[iobs, icell] = neg_sus * tzx
                Tzy[iobs, icell] = neg_sus * tzy
                Tzz[iobs, icell] = neg_sus * tzz

    if ti.static(not compressed):
        if ti.static(write_to_mat):
            for i in range(3 * nC):
                mat[i, i] += 1
        else:
            for i in range(nC):
                Txx[i, i] += 1
                Tyy[i, i] += 1
                Tzz[i, i] += 1


@ti_kernel
def tensor_to_ext_arr(tensor: ti.template(), arr: ti.types.ndarray(),
                      x0: ti.template(), xstride: ti.template(),
                      y0: ti.template(), ystride: ti.template()):
    for I in ti.grouped(tensor):
        arr[I[0] * xstride + x0, I[1] * ystride + y0] = tensor[I]


@ti_func
def ti_use(_):
    pass


@ti_func
def multiply_with_row_stride_3(
        out: ti.template(),
        A: ti.template(),
        B: ti.template(),
        C: ti.template(),
        v: ti.template(),
        row0: ti.template(), row_stride: ti.template()
):
    for r in range(A.shape[0]):
        row = r * row_stride + row0

        summed = (
                A[r, 0] * v[0, 0]
                + B[r, 0] * v[1, 0]
                + C[r, 0] * v[2, 0]
        )

        for c in range(1, A.shape[1]):
            col = c * 3
            summed += (
                    A[r, c] * v[col + 0, 0]
                    + B[r, c] * v[col + 1, 0]
                    + C[r, c] * v[col + 2, 0]
            )

        out[row, 0] = summed


@ti_func
def multiply_with_row_stride_3_compressed(
        out: ti.template(),
        indices_mat: ti.template(),
        A: ti.template(),
        B: ti.template(),
        C: ti.template(),
        v: ti.template(),
        row0: ti.template(), row_stride: ti.template()
):
    identity = ti.Matrix([[0, 0, 0], [0, 0, 0]])
    identity[1, row0] = 1

    for r in range(indices_mat.shape[0]):
        row = r * row_stride + row0

        p = 0
        if r == 0:
            p = 1
        ind = indices_mat[r, 0]
        summed = (
            (A[ind] + identity[p, 0]) * v[0, 0]
            + (B[ind] + identity[p, 1]) * v[1, 0]
            + (C[ind] + identity[p, 2]) * v[2, 0]
        )

        for c in range(1, indices_mat.shape[1]):
            p = 0
            if r == c:
                p = 1
            ind = indices_mat[r, c]
            col = c * 3
            summed += (
                (A[ind] + identity[p, 0]) * v[col + 0, 0]
                + (B[ind] + identity[p, 1]) * v[col + 1, 0]
                + (C[ind] + identity[p, 2]) * v[col + 2, 0]
            )

        out[row, 0] = summed
