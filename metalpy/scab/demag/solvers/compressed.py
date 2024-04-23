import math
import warnings
from typing import Union, Literal

import numpy as np
import taichi as ti
from taichi.lang.misc import is_extension_supported
from taichi.lang.util import to_taichi_type

from metalpy.utils.numeric import limit_significand
from metalpy.utils.taichi import ti_kernel, ti_field, ti_FieldsBuilder, ti_func, ti_size_t, ti_size_dtype, \
    ti_pyfunc
from metalpy.utils.ti_solvers import matrix_free
from .demag_solver_context import DemagSolverContext
from .kernel import kernel_matrix_forward
from .solver import DemagnetizationSolver


class CompressedSolver(DemagnetizationSolver):
    Optimal = 'optimal'
    Strict = 'strict'

    def __init__(
            self,
            context: DemagSolverContext,
            compressed_size: Union[int, float, None] = None,
            deterministic: Union[bool, str] = True,
            quantized: bool = False,
            symmetric: Union[bool, Literal['strict']] = False
    ):
        """该函数将核矩阵通过哈希表进行压缩，以计算效率为代价换取大幅降低内存需求量

        Parameters
        ----------
        context
            退磁求解上下文
        compressed_size
            压缩表尺寸，
            如果为整数，则用于指定尺寸大小，
            如果为浮点数则设置为核矩阵大小的对应比例，
            默认None则为1 / 32
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

        Notes
        -----
        优势：

        - 可以大幅降低内存需求量，对规则网格效果尤其明显，规则网格上约可以将内存需求降低到千分之一

        缺陷：

        - 网格规模仍然受到taichi的int32索引限制
        - 对计算效率有较大影响
        - 由于索引矩阵的存在，实际上仍然为O(n^2)的空间复杂度，只是常数项较小

        `deterministic` 指定是否采用确定性模式并行哈希，可选的取值为：

        - `True` （默认值），确保核矩阵正确，但会牺牲一定计算效率和空间
        - `False` 则会抛弃确定性约束，牺牲一定精度获取更优的时空间效率
        - `CompressedForward.Optimal` 时直接采用unique函数计算理想压缩结果

        `quantized` 指定是否对索引进行量化压缩，将一个整形拆分为多个值从而实现压缩，可选的取值为：

        - `False` 不启用压缩（默认值），采用默认索引类型存储（ti_size_t）
        - `True` 启用压缩，使用更小位宽来存储索引，从而在相同的空间中存储更多索引值，减少一定的空间需求

            在当前64位位宽下，只支持最多压缩到21位，索引约200万项，
            更宽则一个int64中只能存储2个索引，和32位相同，此时自动转为非压缩模式。

        `symmetric` 指定是否启用对称模式（当网格为规则网格时可以启用）存储索引矩阵，
        对称模式下，所有对角线元素值相同，存储在 indices_mat[0] ，上三角部分逐行存储在 indices_mat[1:] 。

        注意事项：

        - 网格尺寸建议采用 `IEEE754` 可以精确表示的值，防止因计算误差干扰压缩效果
        - 例如使用 0.375 或 0.4375 替代 0.4
        - 可以使用 `metalpy.utils.numeric.limit_significand` 来搜索合适的网格尺寸
        """
        super().__init__(context)

        self.deterministic = deterministic

        # 检查二进制小数的稳定性，例如 `0.4` 无法使用二进制小数精确表示，因此在其上的加减法会引入误差
        # 而CompressedSolver依赖数值的绝对相等来实现去重，对这种误差较为敏感，需要进行检查
        check_binary_floats_stability(
            self.receiver_locations,
            self.kernel_dtype,
            np.max(self.n_cells_on_each_axes)
        )

        default_table_size = get_default_table_size(self.n_cells)
        if compressed_size is None:
            compressed_size = default_table_size

        if np.issubdtype(type(compressed_size), np.floating):
            compressed_size = int(default_table_size * compressed_size)

        self.compressed_size = compressed_size
        self.used, self.overflow = 0, False

        self.builder = builder = ti_FieldsBuilder()

        self.Tmat321 = [
            [ti_field(self.kernel_dtype) for _ in range(3)],  # Txx, Txy, Txz
            [ti_field(self.kernel_dtype) for _ in range(2)],  # ---- Tyy, Tyz
            [ti_field(self.kernel_dtype) for _ in range(1)],  # --------- Tzz
        ]
        self.Tmat6 = [t for ts in self.Tmat321 for t in ts]

        builder.dense(ti.i, compressed_size).place(*self.Tmat6)

        self.symmetric, axes, indices_size = self._create_indices_specs(symmetric=symmetric)
        self.quantized, self.indices_mat = self._build_indices(axes, indices_size, quantized=quantized)

        builder.finalize()

    def build_kernel(self, model):
        self.used, self.overflow = compress_kernel(
            self.Tmat6,
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.indices_mat,
            deterministic=self.deterministic,
            symmetric=self.symmetric,
            cutoff=self.cutoff
        )

        if not np.allclose(model, model[0]):
            warnings.warn(
                '`CompressedSolver` does not support non-uniform susceptibility model yet.'
                ' May lead to unexpected result.'
            )

        kernel_matrix_forward(
            self.receiver_locations,
            self.xn, self.yn, self.zn, model,
            *self.Tmat6, mat=np.empty(0), kernel_dtype=self.kernel_dt,
            write_to_mat=False, compressed=True,
            apply_susc_model=True  # TODO: 考虑非均匀磁化率的情况
        )

    def solve(self, magnetization, model):
        solver = not self.symmetric and solve_Tx_b_compressed or solve_Tx_b_compressed_symmetric
        return solver(self.Tmat321, self.indices_mat, magnetization, progress=self.progress)

    def _create_indices_specs(self, symmetric: Union[bool, Literal['strict']]):
        is_symmetric = self.context.is_symmetric
        if symmetric is None:
            symmetric = not self.is_cpu and is_symmetric
        elif symmetric and not is_symmetric:
            if symmetric == CompressedSolver.Strict:
                raise ValueError(
                    '`symmetric` mode is not allowed on a non-symmetric mesh.'
                    ' Try explicitly setting `symmetric=True` to force using symmetric mode.'
                )
            else:
                warnings.warn(
                    '`symmetric` mode is enabled on a non-symmetric mesh,'
                    ' which may lead to unexpected result.'
                )

        if symmetric == CompressedSolver.Strict:
            symmetric = True

        if not symmetric:
            axes = ti.ij
            indices_size = [self.n_obs, self.n_cells]
        else:
            axes = ti.i
            indices_size = [symmetric_mat_size(self.n_obs, self.n_cells)]

        return symmetric, axes, indices_size

    def _build_indices(self, axes, indices_size, quantized):
        should_warn = quantized is not None  # 如果以默认选项形式传入，则在忽略量化选项时不产生警告信息
        quantizable = is_extension_supported(self.arch, ti.extension.quant)

        if quantized is None:
            quantized = not self.is_cpu and quantizable
        elif quantized is True and not quantizable:
            warnings.warn(f'`{self.arch.name}` does not support quantized types.'
                          f' Ignoring `quantized` and using {ti_size_dtype.__name__} instead.')

        self.quantized = quantized

        indices_mat = None
        while quantized:
            # TODO: 万一将来位宽变成128了？
            primitive_bits = np.iinfo(np.int64).bits
            quantized_bits = math.ceil(np.log2(self.compressed_size))
            n_packed = primitive_bits // quantized_bits
            if n_packed <= 2:
                if should_warn:
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
            indices_size[-1] = math.ceil(indices_size[-1] / n_packed)

            (
                self.builder
                .dense(axes, indices_size)
                .quant_array(axes[-1:], n_packed, max_num_bits=primitive_bits)
                .place(indices_mat)
            )

            break

        if indices_mat is None:
            indices_mat = ti_field(ti_size_t)
            self.builder.dense(axes, indices_size).place(indices_mat)

        return quantized, indices_mat


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
        deterministic,
        symmetric,
        cutoff
):
    if deterministic not in [True, False]:
        assert deterministic == CompressedSolver.Optimal

    table_size = Tmat6[0].shape[0]
    default_table_size = get_default_table_size(xn.shape[0])

    if deterministic == CompressedSolver.Optimal:
        used, overflow = compress_kernel_optimal(
            Tmat6, receiver_locations,
            xn, yn, zn,
            indices_mat,
            symmetric,
            cutoff
        )
    else:
        used, overflow = compress_kernel_by_hash(
            Tmat6, receiver_locations,
            xn, yn, zn,
            indices_mat,
            deterministic,
            symmetric,
            cutoff
        )

    if overflow:
        raise RuntimeError(f'Compression hash table overflowed.'
                           f' Consider using larger `compressed_size` like'
                           f' `{int(table_size * 1.5)}` or'
                           f' `{table_size / default_table_size * 1.5:.2f}`'
                           f' (currently {table_size}).')

    memory_efficiency = used / table_size
    if memory_efficiency < 0.95:
        # 一个特点是，尤其是对于规则六边形网格
        # 不同网格关系数的和有效网格数近似线性关系
        warnings.warn(f'Low memory efficiency detected ({memory_efficiency:.2%}).'
                      f' Consider setting `compressed_size` to'
                      f' `{int(used / 0.95)}`'
                      f' or'
                      f' `{used / 0.95 / default_table_size:.4f}`'
                      f'.')

    return used, overflow > 0


def compress_kernel_optimal(
        mats,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        indices_mat,
        symmetric: bool,
        cutoff: float
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

        inverse = inverse.reshape(table.shape[:2]).astype(ti_size_dtype)
        if not symmetric:
            indices_mat.from_numpy(inverse)
        else:
            indices_mat.from_numpy(inverse[np.triu_indices(xn.shape[0])])

    return used, overflow > 0


def compress_kernel_by_hash(
        mats,
        receiver_locations: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
        zn: np.ndarray,
        indices_mat,
        deterministic: bool,
        symmetric: bool,
        cutoff: float
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
        invalid_value_for_dx1,
        symmetric=symmetric,
        cutoff=cutoff
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
        invalid_value_for_dx1: ti.template(),
        symmetric: ti.template(),
        cutoff: ti.template()
):
    for I in ti.grouped(Txx):
        Txx[I] = ti.math.inf

    nC = xn.shape[0]
    nObs = receiver_locations.shape[0]
    table_size = Txx.shape[0]
    overflow[0] = ti.i8(0)
    qb = total_bits // 4  # quarter bits

    # 初始化保留元素区
    # 设置位置 0 为 0 元素，在启用截断距离时，超出截断距离的网格间影响统一压缩为0
    faraway = 0  # 极远距离点
    reserved: ti_size_t = 0  # 保留元素个数
    if ti.static(cutoff != ti.math.inf):
        # 虽然似乎没有必要，但还是显式声明一下
        reserved += 1
        # kernel_matrix_forward 检测 Txx 为 inf 会视为无穷距离
        Txx[faraway] = Txy[faraway] = Txz[faraway] = Tyy[faraway] = Tyz[faraway] = Tzz[faraway] = ti.math.inf
    table_size -= reserved

    total = nObs * nC
    if ti.static(symmetric):
        total = symmetric_mat_size(nObs, nC)

    for i in ti.ndrange(total):
        if overflow[0] > 0:
            continue

        iobs, icell = i // nC, i % nC
        if ti.static(symmetric):
            iobs, icell = index_triu2mat(i, nC)

        dz2 = zn[icell, 1] - receiver_locations[iobs, 2]
        dz1 = zn[icell, 0] - receiver_locations[iobs, 2]
        dy2 = yn[icell, 1] - receiver_locations[iobs, 1]
        dy1 = yn[icell, 0] - receiver_locations[iobs, 1]
        dx2 = xn[icell, 1] - receiver_locations[iobs, 0]
        dx1 = xn[icell, 0] - receiver_locations[iobs, 0]

        a = table_size

        if ti.static(cutoff != ti.math.inf):
            dist = ti.sqrt(
                (dx1 + dx2) ** 2
                + (dy1 + dy2) ** 2
                + (dz1 + dz2) ** 2
            ) / 2

            if dist > cutoff:
                a = faraway

        if a == table_size:
            if ti.static(not symmetric):
                # 由于互换对称性，核矩阵会存在一定的对称性
                # 极端情况是规则网格，此时核矩阵为对称矩阵，启用对称模式即不再需要该trick

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
            ) % table_size + reserved

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
                if a >= table_size + reserved:
                    a -= table_size

                if a == a0:
                    # 无法解决冲突，认为哈希表已经溢出
                    overflow[0] = ti.i8(1)
                    break

            # 容量超了的话，其实已经不需要管下面数据覆盖的问题了
            # if overflow[0] > 0:
            #     # print('寄！')
            #     continue
            Txx[a] = ti.cast(dx1, Txx.dtype)
            Txy[a] = ti.cast(dx2, Txy.dtype)
            Txz[a] = ti.cast(dy1, Txz.dtype)
            Tyy[a] = ti.cast(dy2, Tyy.dtype)
            Tyz[a] = ti.cast(dz1, Tyz.dtype)
            Tzz[a] = ti.cast(dz2, Tzz.dtype)

        if ti.static(not symmetric):
            indices_mat[iobs, icell] = a
        else:
            indices_mat[i] = a


@ti_func
def cshift(key, b):
    return (key << b) | (key >> b)


def solve_Tx_b_compressed(Tmat321, indices_mat, m, progress: bool = False):
    dtype = to_taichi_type(m.dtype)

    @ti_func
    def extract(r, c):
        ind = indices_mat[r, c]
        return (
            Tmat321[0][0][ind], Tmat321[0][1][ind], Tmat321[0][2][ind],
            Tmat321[1][0][ind], Tmat321[1][1][ind],
            Tmat321[2][0][ind]
        )

    @ti_kernel
    def linear(x: ti.template(), Ax: ti.template()):
        n_obs = Ax.shape[0] // 3
        n_cells = x.shape[0] // 3

        for r in range(n_obs):
            row = r * 3

            bx: dtype = 0
            by: dtype = 0
            bz: dtype = 0

            for c in range(0, n_cells):
                col = c * 3

                txx, txy, txz, tyy, tyz, tzz = extract(r, c)
                mx, my, mz = x[col + 0], x[col + 1], x[col + 2]

                bx += txx * mx + txy * my + txz * mz
                by += txy * mx + tyy * my + tyz * mz
                bz += txz * mx + tyz * my + tzz * mz

            Ax[row + 0] = bx
            Ax[row + 1] = by
            Ax[row + 2] = bz

    return matrix_free.cg(ti.linalg.LinearOperator(linear), m, progress=progress).to_numpy()


def solve_Tx_b_compressed_symmetric(Tmat321, indices_mat, m, progress: bool = False):
    dtype = to_taichi_type(m.dtype)
    n_cells = m.shape[0] // 3

    @ti_func
    def extract_by_index(i):
        ind = indices_mat[i]
        return (
            Tmat321[0][0][ind], Tmat321[0][1][ind], Tmat321[0][2][ind],
            Tmat321[1][0][ind], Tmat321[1][1][ind],
            Tmat321[2][0][ind]
        )

    @ti_func
    def extract(r, c):
        i = index_mat2triu(r, c, n_cells)
        return extract_by_index(i)

    @ti_func
    def extract_diag():
        return extract_by_index(0)

    @ti_kernel
    def linear(x: ti.template(), Ax: ti.template()):
        n_obs = Ax.shape[0] // 3
        n_cells = x.shape[0] // 3

        # 对角线和上三角部分
        txx_, txy_, txz_, tyy_, tyz_, tzz_ = extract_diag()
        for r in range(n_obs):
            row = r * 3

            mx_, my_, mz_ = x[row + 0], x[row + 1], x[row + 2]

            bx_u: dtype = txx_ * mx_ + txy_ * my_ + txz_ * mz_
            by_u: dtype = txy_ * mx_ + tyy_ * my_ + tyz_ * mz_
            bz_u: dtype = txz_ * mx_ + tyz_ * my_ + tzz_ * mz_

            for c in range(r + 1, n_cells):
                col = c * 3

                txx, txy, txz, tyy, tyz, tzz = extract(r, c)
                mx, my, mz = x[col + 0], x[col + 1], x[col + 2]

                bx_u += txx * mx + txy * my + txz * mz
                by_u += txy * mx + tyy * my + tyz * mz
                bz_u += txz * mx + tyz * my + tzz * mz

            Ax[row + 0] = bx_u
            Ax[row + 1] = by_u
            Ax[row + 2] = bz_u

        # 下三角部分
        for c in range(1, n_cells):
            col = c * 3

            bx_l: dtype = 0
            by_l: dtype = 0
            bz_l: dtype = 0

            for r in range(c):
                row = r * 3

                txx, txy, txz, tyy, tyz, tzz = extract(r, c)
                mx, my, mz = x[row + 0], x[row + 1], x[row + 2]

                bx_l += txx * mx + txy * my + txz * mz
                by_l += txy * mx + tyy * my + tyz * mz
                bz_l += txz * mx + tyz * my + tzz * mz

            Ax[col + 0] += bx_l
            Ax[col + 1] += by_l
            Ax[col + 2] += bz_l

    return matrix_free.cg(ti.linalg.LinearOperator(linear), m, progress=progress).to_numpy()


@ti_kernel
def copy_col_vector_from_matrix(
        dst: ti.template(), src: ti.types.ndarray(),
        col: ti.template()
):
    size = ti.min(dst.shape[0], src.shape[0])
    for i in range(size):
        dst[i] = src[i, col]


def check_binary_floats_stability(
        receiver_locations, kernel_dtype, max_cells_along_axes, threshold=0.95, axes_names='XYZ'
):
    """检查二进制小数的稳定性，例如 `0.4` 无法使用二进制小数精确表示，
    因此在其上的运算会引入误差，误差累积后会导致结果发生细微的偏差

    Parameters
    ----------
    receiver_locations
        观测点坐标
    kernel_dtype
        核矩阵数据类型
    max_cells_along_axes
        单个坐标轴上的最大网格数
    threshold
        稳定性阈值（0 ~ 1，列元素中稳定的元素数大于该比例则认为合格）
    axes_names
        用于输出的坐标轴名
    """
    # 检查乘积结果是否在 `kernel_dtype` 精度范围内
    digits = np.finfo(kernel_dtype).precision
    relations = receiver_locations - receiver_locations[0]  # 网格间关系，即 网格长度 乘以 间隔网格数
    stable_mask = np.equal(relations, limit_significand(relations, tol=10 ** -digits))
    axes_ratio = np.count_nonzero(stable_mask, axis=0) / stable_mask.shape[0]

    # 推算乘以 网格数 导致额外消耗的有效位数上界
    required_digits = np.log10(max_cells_along_axes)
    tol_hint = 10 ** -(digits - required_digits)

    for axis, rate in enumerate(axes_ratio):
        if rate < threshold:
            if axes_names is not None:
                axis = axes_names[axis]
            warnings.warn(f'Low numeric stability detected on axis {axis}'
                          f' ({rate:.2%} < {threshold:.2%}),'
                          f' which may severely affect compression rate.'
                          f' Consider adjusting mesh cell sizes by'
                          f' `limit_significand(<your_cell_sizes>, tol={tol_hint:.4})`.'
                          f'\n'
                          f' See `metalpy.utils.numeric.limit_significand`'
                          f' for more details.')


def get_default_table_size(n_cells):
    return 100 * n_cells


@ti_pyfunc
def mul_and_div2(a, b):
    """计算 a * b // 2，通过调换除法顺序来防止溢出
    """
    p = a & 1  # a是否是2的倍数
    a >>= 1 - p
    b >>= p

    return a * b


@ti_pyfunc
def symmetric_mat_size(nobs, nc):
    """计算对称模式下的存储大小，包含一个对角线元素和上三角部分
    """
    # nobs * (nc - 1) // 2 + 1
    return mul_and_div2(nobs, nc - 1) + 1


@ti_func
def index_triu2mat(i, n):
    """通过索引号i计算上三角矩阵下的行列号

    i = 0 时代表对角线元素，返回 c = r = 0

    i > 0 时计算对应上三角部分的行列号 c 和 r
    """
    r: ti_size_t = 0
    c: ti_size_t = 0

    if i != 0:
        # ref: https://stackoverflow.com/a/69292749
        # 注意此处 i=0 代表所有对角线元素，因此对应到回答中 k=i-1
        # 但是由于计算量较大且存在精度损失，效率反而不如迭代计算
        # r = n - 2 - ti.math.floor(ti.math.sqrt(4.0 * n * (n - 1) - (8 * (i - 1)) - 7) / 2.0 - 0.5, dtype=ti_size_t)

        r = 0
        c = i
        while c >= n:
            r = r + 1 + (c - (n - 1)) // ((n - 1) - (r + 1))
            c = i - mul_and_div2(2 * n - (r + 1), r) + r
        while c <= r:
            c = c + n - (r + 1)
            r -= 1

    return r, c


@ti_pyfunc
def index_mat2triu(r, c, n):
    """通过上三角矩阵下的行列号计算索引号i，需要调用方保证 c > r
    """
    # n * r - r * (r + 1) // 2 + (c - (r + 1)) + 1
    # 可以变形为
    # (2 * n - (r + 1)) * r // 2 + (c - (r + 1)) + 1
    return mul_and_div2(2 * n - (r + 1), r) + (c - (r + 1)) + 1
