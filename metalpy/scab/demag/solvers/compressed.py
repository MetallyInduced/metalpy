import math
import warnings
from typing import Union

import numpy as np
import taichi as ti
from taichi.lang.misc import is_extension_supported
from taichi.lang.util import to_taichi_type

from metalpy.scab.utils.misc import Field
from metalpy.utils.numeric import limit_significand
from metalpy.utils.taichi import ti_kernel, ti_field, ti_FieldsBuilder, ti_func, copy_from, ti_size_t, ti_size_dtype, \
    ti_cfg
from metalpy.utils.ti_solvers import matrix_free
from .kernel import kernel_matrix_forward
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
            quantized: bool = False,
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
            采用确定性模式并行哈希，默认为True，牺牲一定计算效率和空间确保核矩阵正确。
            指定为CompressedForward.Optimal时直接采用unique函数计算理想压缩结果
        quantized
            采用量化模式压缩核矩阵，默认为False。
            True则启用压缩，使用更小位宽来存储核矩阵，减少一定的空间需求，略微牺牲计算效率。
            （可能会报要求尺寸为2的幂的性能警告）
        progress
            是否输出求解进度条，默认为False不输出

        Notes
        -----
        优势：

        - 可以大幅降低内存需求量，对规则网格效果尤其明显，规则网格上约可以将内存需求降低到千分之一

        缺陷：

        - 网格规模仍然受到taichi的int32索引限制
        - 对计算效率有较大影响

        `deterministic` 指定是否采用确定性模式并行哈希，可选的取值为：

        - `True` （默认值），确保核矩阵正确，但会牺牲一定计算效率和空间
        - `False` 则会抛弃确定性约束，牺牲一定精度获取更优的时空间效率
        - `CompressedForward.Optimal` 时直接采用unique函数计算理想压缩结果

        `quantized` 指定是否对索引进行量化压缩，将一个整形拆分为多个值从而实现压缩，可选的取值为：

        - `False` 不启用压缩（默认值），采用默认索引类型存储（ti_size_t）
        - `True` 启用压缩，使用更小位宽来存储索引，从而在相同的空间中存储更多索引值，减少一定的空间需求

            在当前64位位宽下，只支持最多压缩到21位，索引约200万项，
            更宽则一个int64中只能存储2个索引，和32位相同，此时自动转为非压缩模式。

        注意事项：

        - 网格尺寸建议采用 `IEEE754` 可以精确表示的值，防止因计算误差干扰压缩效果
        - 例如使用 0.375 或 0.4375 替代 0.4
        - 可以使用 `metalpy.utils.numeric.limit_significand` 来搜索合适的网格尺寸
        """
        super().__init__(receiver_locations, xn, yn, zn, base_cell_sizes, source_field)

        # 检查二进制小数的稳定性，例如 `0.4` 无法使用二进制小数精确表示，因此在其上的加减法会引入误差
        # 而CompressedSolver依赖数值的绝对相等来实现去重，对这种误差较为敏感，需要进行检查
        check_binary_floats_stability(
            receiver_locations - receiver_locations[0],
            threshold=0.95,
            axes_names='XYZ'
        )

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

        self.Tmat321 = [
            [ti_field(self.kernel_type) for _ in range(3)],  # Txx, Txy, Txz
            [ti_field(self.kernel_type) for _ in range(2)],  # ---- Tyy, Tyz
            [ti_field(self.kernel_type) for _ in range(1)],  # --------- Tzz
        ]
        self.Tmat33 = [
            self.Tmat321[0],  # Txx, Txy, Txz
            [self.Tmat321[0][1]] + self.Tmat321[1],  # Tyx, Tyy, Tyz
            [self.Tmat321[0][2], self.Tmat321[1][1]] + self.Tmat321[2]  # Tzx, Tzy, Tzz
        ]
        self.Tmat6 = [t for ts in self.Tmat321 for t in ts]

        builder.dense(ti.i, compressed_size).place(*self.Tmat6)

        indices_mat = None
        while quantized:
            arch = ti_cfg().arch
            if not is_extension_supported(arch, ti.extension.quant):
                warnings.warn(f'`{arch.name}` does not support quantized types.'
                              f' Ignoring `quantized` and using {ti_size_dtype.__name__} instead.')
                break

            # TODO: 万一将来位宽变成128了？
            primitive_bits = np.iinfo(np.int64).bits
            quantized_bits = math.ceil(np.log2(compressed_size))
            n_packed = primitive_bits // quantized_bits
            if n_packed <= 2:
                warnings.warn(f'A single `uint{primitive_bits}` can pack only'
                              f' {n_packed} quantized index type `uint{quantized_bits}`.'
                              f' Ignoring `quantized` and using `{ti_size_dtype.__name__}` instead.')
                break

            indices_type = ti.types.quant.int(
                bits=quantized_bits,
                signed=False,
                compute=ti_size_t
            )
            indices_mat = ti_field(indices_type)
            (
                builder
                .dense(ti.ij, (nObs, math.ceil(nC / n_packed)))
                .quant_array(ti.j, n_packed, max_num_bits=primitive_bits)
                .place(indices_mat)
            )
            break

        if indices_mat is None:
            indices_mat = ti_field(ti_size_t)
            builder.dense(ti.ij, (nObs, nC)).place(indices_mat)

        self.indices_mat = indices_mat

        builder.finalize()

    def build_kernel(self, model):
        self.used, self.overflow = compress_kernel(
            self.Tmat6,
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.indices_mat,
            deterministic=self.deterministic
        )

        kernel_matrix_forward(
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.base_cell_sizes, model,
            *self.Tmat6, np.empty(0),
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
        Tmat6,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        indices_mat,
        deterministic
):
    if deterministic not in [True, False]:
        assert deterministic == CompressedSolver.Optimal

    table_size = Tmat6[0].shape[0]
    kernel_size = np.prod(indices_mat.shape)

    if deterministic == CompressedSolver.Optimal:
        used, overflow = compress_kernel_optimal(
            Tmat6, receiver_locations,
            xn, yn, zn,
            indices_mat
        )
    else:
        used, overflow = compress_kernel_by_hash(
            Tmat6, receiver_locations,
            xn, yn, zn,
            indices_mat,
            deterministic
        )

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


def compress_kernel_optimal(
        mats,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        indices_mat
):
    n_param = len(mats)
    table_size = mats[0].shape[0]

    table = np.c_[xn, yn, zn][:, None, :] - np.repeat(receiver_locations, 2, axis=1)[None, :, :]
    compressed, inverse = np.unique(table.reshape(-1, n_param), axis=0, return_inverse=True)

    used = compressed.shape[0]
    overflow = used > table_size

    if not overflow:
        for i in range(n_param):
            copy_col_vector_from_matrix(mats[i], compressed, i)
        indices_mat.from_numpy(inverse.reshape(table.shape[:2]).astype(ti_size_dtype))

    return used, overflow > 0


def compress_kernel_by_hash(
        mats,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        indices_mat,
        deterministic: bool
):
    table_size = mats[0].shape[0]

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
    used = np.count_nonzero(mats[0].to_numpy() < invalid_value_for_dx1)

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
        Tyy: ti.template(),
        Tyz: ti.template(),
        Tzz: ti.template(),
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

        # 由于互换对称性，核矩阵会存在一定的对称性（注：特例是规则网格，此时核矩阵为对称矩阵）
        # 因此通过确保网格关系的唯一性，可以最高将压缩率提高到原来的50%
        # 唯一性条件具体以dx条件为例：
        #   情况1. dx1和dx2同号，保证均为正
        #   情况2. dx1和dx2异号，保证dx1绝对值小于dx2 （注：该情况对于规则网格无意义）
        # 检查过程如下：
        # 1. 检查dx条件，不满足则执行互换
        # 2. 如果dx相反（dx1和dx2互为相反数，此时dx1和dx2互换前后数值不变），则检查dy条件，不满足则执行互换
        # 3. 如果dx和dy均相反，则检查dz条件，不满足则执行互换
        # 提升比需要在CPU模式下验证（对于GPU，由于冲突的存在，无法达到理想提升比）
        e1, e2 = dx1 + dx2 == 0, dy1 + dy2 == 0
        swap_x = dx1 < 0 and -dx1 > dx2  # 检查dx条件
        swap_y = e1 and dy1 < 0 and -dy1 > dy2  # dx相反，检查dy条件
        swap_z = e1 and e2 and dz1 < 0 and -dz1 > dz2  # dx和dy相反，检查dz条件

        if swap_x or swap_y or swap_z:
            dx1, dx2 = -dx2, -dx1
            dy1, dy2 = -dy2, -dy1
            dz1, dz2 = -dz2, -dz1

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
                and Tyy[a] == dy2
                and Tyz[a] == dz1
                and Tzz[a] == dz2
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
        Tyy[a] = dy2
        Tyz[a] = dz1
        Tzz[a] = dz2


@ti_func
def cshift(key, b):
    return (key << b) | (key >> b)


def solve_Tx_b_compressed(Tmat33, indices_mat, m, progress: bool = False):
    with ti_FieldsBuilder() as builder:
        x = builder.place_dense_like(m[:, None])
        b = builder.place_dense_like(m[:, None])

        builder.finalize()

        copy_from(b, m[:, None])

        dtype = x.dtype

        @ti_kernel
        def linear(x: ti.template(), Ax: ti.template()):
            n_obs = indices_mat.shape[0]
            n_cells = x.shape[0] // 3  # 启用quant类型后，indices_mat中可能会有多余的列存在

            for r in range(n_obs):
                summed0: dtype = 0
                summed1: dtype = 0
                summed2: dtype = 0

                for c in range(0, n_cells):
                    ind = indices_mat[r, c]
                    col = c * 3
                    summed0 += (
                            Tmat33[0][0][ind] * x[col + 0, 0]
                            + Tmat33[0][1][ind] * x[col + 1, 0]
                            + Tmat33[0][2][ind] * x[col + 2, 0]
                    )
                    summed1 += (
                            Tmat33[1][0][ind] * x[col + 0, 0]
                            + Tmat33[1][1][ind] * x[col + 1, 0]
                            + Tmat33[1][2][ind] * x[col + 2, 0]
                    )
                    summed2 += (
                            Tmat33[2][0][ind] * x[col + 0, 0]
                            + Tmat33[2][1][ind] * x[col + 1, 0]
                            + Tmat33[2][2][ind] * x[col + 2, 0]
                    )

                Ax[3 * r + 0, 0] = summed0
                Ax[3 * r + 1, 0] = summed1
                Ax[3 * r + 2, 0] = summed2

        matrix_free.cg(ti.linalg.LinearOperator(linear), b, x, progress=progress)

        return x.to_numpy()


@ti_kernel
def copy_col_vector_from_matrix(
        dst: ti.template(), src: ti.types.ndarray(),
        col: ti.template()
):
    size = ti.min(dst.shape[0], src.shape[0])
    for i in range(size):
        dst[i] = src[i, col]


def check_binary_floats_stability(matrix, threshold=0.95, axes_names=None):
    """检查二进制小数的稳定性，例如 `0.4` 无法使用二进制小数精确表示，
    因此在其上的运算会引入误差，误差累积后会导致结果发生细微的偏差

    Parameters
    ----------
    matrix
        需要检查稳定性的矩阵
    threshold
        稳定性阈值（0 ~ 1，列元素中稳定的元素数大于该比例则认为合格）
    axes_names
        用于输出的坐标轴名
    """
    stable_mask = np.equal(matrix, limit_significand(matrix))
    axes_ratio = np.count_nonzero(stable_mask, axis=0) / stable_mask.shape[0]
    for axis, rate in enumerate(axes_ratio):
        if rate < threshold:
            if axes_names is not None:
                axis = axes_names[axis]
            warnings.warn(f'Low numeric stability detected on axis {axis}'
                          f' ({rate:.2%} < {threshold:.2%}),'
                          f' which may severely affect compression rate.'
                          f' Consider adjusting mesh cell sizes by'
                          f' `limit_significand(your_cell_sizes)`.'
                          f'\n'
                          f' See `metalpy.utils.numeric.limit_significand`'
                          f' for more details.')
