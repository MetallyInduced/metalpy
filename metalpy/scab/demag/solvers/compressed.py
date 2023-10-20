import warnings
from typing import Union

import numpy as np
import taichi as ti
from taichi.lang.util import to_taichi_type

from metalpy.scab.utils.misc import Field
from metalpy.utils.taichi import ti_kernel, ti_field, ti_FieldsBuilder, ti_func, copy_from, ti_size_t, ti_size_dtype
from .kernel import kernel_matrix_forward
from .matrix_free import conjugate_gradient
from .solver import DemagnetizationSolver


class CompressedSolver(DemagnetizationSolver):
    Optimal = 'optimal'

    def __init__(
            self,
            receiver_locations: np.ndarray,
            xn: np.ndarray,
            yn: np.ndarray,
            zn: np.ndarray,
            base_cell_sizes: np.ndarray,
            source_field: Field,
            compressed_size: Union[int, float, None] = None,
            deterministic: Union[bool, str] = True,
            progress: bool = False
    ):
        """该函数将核矩阵通过哈希表进行压缩，以计算效率为代价换取大幅降低内存需求量

        Parameters
        ----------
        receiver_locations
            观测点
        xn, yn, zn
            网格边界
        base_cell_sizes
            网格最小单元大小
        compressed_size
            压缩表尺寸，
            如果为整数，则用于指定尺寸大小，
            如果为浮点数则设置为核矩阵大小的对应比例，
            默认None则为1 / 32
        deterministic
            采用确定性模式并行哈希，默认为True，确保核矩阵正确，但会牺牲一定计算效率和空间。
            False则会抛弃确定性约束，牺牲一定精度获取更优的时空间效率。
            指定为CompressedForward.Optimal时直接采用unique函数计算理想压缩结果
        progress
            是否输出求解进度条，默认为False不输出

        Notes
        -----
        优势：
        1. 可以大幅降低内存需求量，对规则网格效果尤其明显，可以将内存需求降低上千倍

        缺陷：
        1. 网格规模仍然受到taichi的int32索引限制
        2. 对计算效率有较大影响
        3. 考虑性能使用了并发哈希，可能会导致核矩阵中极小部分数据被覆盖导致丢失
        """
        super().__init__(receiver_locations, xn, yn, zn, base_cell_sizes, source_field)
        self.deterministic = deterministic
        self.progress = progress

        nC = xn.shape[0]
        nObs = receiver_locations.shape[0]

        if compressed_size is None:
            compressed_size = 100000

        if np.issubdtype(type(compressed_size), np.floating):
            compressed_size = int(nC * nObs * compressed_size)

        self.compressed_size = compressed_size
        self.used, self.overflow = 0, False

        self.builder = builder = ti_FieldsBuilder()
        # [[Txx, Txy, Txz], [Tyx, Tyy, Tyz], [Tzx, Tzy, Tzz]]
        self.Tmat33 = [
            [ti_field(self.kernel_type) for _ in range(3)]
            for _ in range(3)
        ]
        self.Tmat9 = [t for ts in self.Tmat33 for t in ts]
        self.indices_mat = ti_field(ti_size_t)  # TODO: taichi当前的索引类型

        # 保证 Tx[xyz]，Ty[xyz]，Tz[xyz] 分别在空间上连续，提高空间近邻性
        for m in self.Tmat33:
            builder.dense(ti.i, compressed_size).place(*m)
        builder.dense(ti.ij, (nObs, nC)).place(self.indices_mat)

        builder.finalize()

    def build_kernel(self, model):
        self.used, self.overflow = compress_kernel(
            self.Tmat33,
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.indices_mat,
            deterministic=self.deterministic
        )

        kernel_matrix_forward(
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.base_cell_sizes, model,
            *self.Tmat9, np.empty(0),
            write_to_mat=False, compressed=True
        )

    def solve(self, magnetization):
        return solve_Tx_b_compressed(self.Tmat33, self.indices_mat, magnetization, progress=self.progress)


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
    寻找哈希表的"好质数"： https://planetmath.org/goodhashtableprimes
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
        indices_mat,
        deterministic
):
    if deterministic not in [True, False]:
        assert deterministic == CompressedSolver.Optimal

    mats = [*Tmat33[0], *Tmat33[1]]
    n_param = len(mats)
    table_size = Tmat33[0][0].shape[0]
    kernel_size = np.prod(indices_mat.shape)

    if deterministic == CompressedSolver.Optimal:
        table = np.c_[xn, yn, zn][:, None, :] - np.repeat(receiver_locations, 2, axis=1)[None, :, :]
        compressed, inverse = np.unique(table.reshape(-1, n_param), axis=0, return_inverse=True)

        used = compressed.shape[0]
        overflow = used > table_size

        if not overflow:
            for i in range(n_param):
                copy_col_vector_from_matrix(mats[i], compressed, i)
            indices_mat.from_numpy(inverse.reshape(table.shape[:2]).astype(ti_size_dtype))
    else:
        # TODO: 等taichi更新获取变量类型的泛型操作后可以简化
        val_type = np.result_type(receiver_locations, xn)
        assert np.issubdtype(val_type, np.floating)
        val_type = getattr(np, str(val_type).replace('float', 'uint'))
        total_bits = np.iinfo(val_type).bits

        overflow = ti_field(ti.i8, 1)
        prime = search_prime(table_size)

        if deterministic:
            # 使用原子操作来防止哈希表的单个条目被多个线程操作，确保核矩阵参数正确
            # 但会导致额外的计算和存储开销，实际压缩后尺寸会大于等于理想压缩结果
            #   GPU下原子操作可能会导致相同参数被存在多个条目下，一般结果大于理想压缩结果
            #   CPU下原子操作没有问题，一般结果等于理想压缩结果
            invalid_value_for_dx1 = 2 * (np.max(receiver_locations[:, 0]) - np.min(xn[:, 0]))
        else:
            # 小于0时hash_unique会取消采用原子操作，允许并行访问操作哈希表
            # 导致核矩阵部分参数产生偏差，实际压缩后尺寸会小于理想压缩结果
            invalid_value_for_dx1 = -1

        hash_unique(
            receiver_locations,
            xn, yn, zn,
            *mats,
            indices_mat,
            to_taichi_type(val_type),
            total_bits,
            overflow,
            prime,
            invalid_value_for_dx1
        )

        overflow = overflow.to_numpy()[0]
        used = np.count_nonzero(Tmat33[0][0].to_numpy() < invalid_value_for_dx1)

    if overflow:
        raise RuntimeError(f'Compression hash table overflowed.'
                           f' Consider using a larger `compressed_size` (currently {table_size}).')

    memory_efficiency = used / table_size
    if memory_efficiency < 0.95:
        # 一个特点是，尤其是对于规则六边形网格
        # 随着网格数增加，不同网格关系数的增速比核矩阵的增速慢
        # 即随着网格数增加，压缩比会越来越低
        warnings.warn(f'Low memory efficiency detected ({memory_efficiency:.2%}).'
                      f' Consider setting `compressed_size` to'
                      f' `{int(used / 0.95)}`'
                      f' or'
                      f' `1 / {int(1 / (used / 0.95 / kernel_size))}`'
                      f'.')

    return used, overflow > 0


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
        total_bits: ti.template(),
        overflow: ti.template(),
        prime: ti.template(),
        invalid_value_for_dx1: ti.template()
):
    for I in ti.grouped(Txx):
        Txx[I] = ti.math.inf

    nC = xn.shape[0]
    nObs = receiver_locations.shape[0]
    table_size = Txx.shape[0]
    overflow[0] = ti.i8(0)
    qb = total_bits // 4  # quarter bits

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
            cshift(p1, qb * 0)
            ^ cshift(p4, qb * 1)
            ^ cshift(p2, qb * 2)
            ^ cshift(p5, qb * 3)
            ^ cshift(p3, qb * 0)
            ^ cshift(p0, qb * 1)
        ) % table_size

        # 预设了网格数大于并行线程数，因此同时查询的键几乎不会重复
        # 访问冲突的概率只取决于哈希函数的碰撞率
        a = a0
        while True:
            if ti.static(invalid_value_for_dx1 > 0):
                # 严格模式，严格防止数据竞争
                # 1. 有可能导致相同参数占用多个哈希条目，导致额外内存开销
                # 2. 依赖原子操作，产生额外计算开销
                # TODO: 替换为ti.atomic_cas
                #   https://github.com/taichi-dev/taichi/issues/1805
                if ti.math.isinf(ti.atomic_min(Txx[a], invalid_value_for_dx1)):
                    break
            else:
                # 非严格模式，有可能因并发修改哈希表条目，导致少量网格参数覆盖
                if ti.math.isinf(Txx[a]):
                    break

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
                # 无法解决冲突，认为哈希表已经溢出
                overflow[0] = ti.i8(1)
                break

        # 容量超了的话，其实已经不需要管下面数据错误的问题了
        # if overflow[0] > 0:
        #     # print('寄！')
        #     continue

        indices_mat[iobs, icell] = a

        Txx[a] = dx1
        Txy[a] = dx2
        Txz[a] = dy1
        Tyx[a] = dy2
        Tyy[a] = dz1
        Tyz[a] = dz2


@ti_func
def cshift(key, b):
    return (key << b) | (key >> b)


def solve_Tx_b_compressed(Tmat33, indices_mat, m, progress: bool = False):
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

        conjugate_gradient(ti.linalg.LinearOperator(linear), b, x, progress=progress)

        return x.to_numpy()


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


@ti_kernel
def copy_col_vector_from_matrix(
        dst: ti.template(), src: ti.types.ndarray(),
        col: ti.template()
):
    size = ti.min(dst.shape[0], src.shape[0])
    for i in range(size):
        dst[i] = src[i, col]
