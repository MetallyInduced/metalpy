import warnings

import numpy as np
import taichi as ti

from metalpy.utils.taichi import ti_kernel
from metalpy.utils.ti_solvers.solver_progress import SolverProgress
from metalpy.utils.type import requires_package
from .demag_solver_context import DemagSolverContext
from .kernel import kernel_matrix_forward_separated
from .solver import DemagnetizationSolver


class BCESolver(DemagnetizationSolver):
    # 不同Toeplitz维度上核矩阵的对称性，为True代表其为负对称，否则为对称
    # 对应维度 0, 1, 2 (z, y, x)
    BlockSkewSymmetric = [
        [False, False, False], [False, True, True], [True, False, True],  # Txx, Txy, Txz
        [False, False, False], [True, True, False],  # Tyy, Tyz
        [False, False, False]  # Tzz
    ]

    def __init__(
            self,
            context: DemagSolverContext
    ):
        """该函数通过BCE（Block Circulant Extension）
        将核矩阵转换为块循环矩阵（BCCB，Block-Circulant-Circulant-Block）形式，
        引入快速傅里叶变换加速退磁矩阵求解过程。

        均匀规则网格上的退磁核矩阵为3级嵌套BTTB矩阵，

        Parameters
        ----------
        context
            退磁求解上下文

        Notes
        -----
        优势：

        - 可以大幅降低内存需求量至O(n)
        - 计算效率最高

        缺陷：

        - 只支持在规则网格上使用
        - 空间复杂度常数较大，相较于其它压缩求解器，可能某些情况下会内存不足

        References
        ----------
        [1] Qiang J, Zhang W, Lu K, et al. A fast forward algorithm for three-dimensional magnetic anomaly on undulating
         terrain[J]. Journal of Applied Geophysics, 2019, 166: 33-41.

        [2] Hogue J D, Renaut R A, Vatankhah S. A tutorial and open source software for the efficient evaluation of
        gravity and magnetic kernels[J]. Computers & Geosciences, 2020, 144: 104575.
        """
        super().__init__(context)

        assert context.is_symmetric, f'{BCESolver.__name__} requires symmetric mesh.'

        self.base_shape = np.ascontiguousarray(self.shape_cells[::-1])  # 原始网格尺寸倒序
        self.bccb_shape = self.base_shape * 2 - 1  # 完整BCCB参数矩阵尺寸
        self.f_bccb_shape = self.bccb_shape.copy()  # 离散傅里叶变换得到的BCCB特征值矩阵尺寸
        self.f_bccb_shape[-1] = (self.bccb_shape[-1] + 1) // 2  # 由于是实矩阵，因此最后维度一定对称，rfftn可以省略负半轴频率

        self._use_torch(not self.is_cpu)

        ops = self.ops

        # 默认填充为 inf ，方便 kernel_matrix_forward 识别填充元素并设置为0
        self.Tensor321 = [
            [ops.array(self.f_bccb_shape, fill=np.inf) for _ in range(3)],  # Txx, Txy, Txz
            [ops.array(self.f_bccb_shape, fill=np.inf) for _ in range(2)],  # ---- Tyy, Tyz
            [ops.array(self.f_bccb_shape, fill=np.inf) for _ in range(1)],  # --------- Tzz
        ]

    @property
    def Tensor6(self):
        return [t for ts in self.Tensor321 for t in ts]

    @property
    def Tmat6(self):
        return [self.ops.flatten(t) for t in self.Tensor6]

    @property
    def Tensor33(self):
        return [
            self.Tensor321[0],
            [self.Tensor321[0][1]] + self.Tensor321[1],
            [self.Tensor321[0][2], self.Tensor321[1][1]] + self.Tensor321[2],
        ]

    @property
    def using_torch(self):
        return self._using_torch

    @property
    def ops(self):
        return self._ops

    def _use_torch(self, using_torch):
        self._using_torch = False

        if using_torch:
            try:
                from .bce_ops.torch_ops import TorchOps as Ops
                self._using_torch = True
            except ImportError:
                requires_package(
                    pkg_name='PyTorch',
                    reason=f"`{BCESolver.__name__}` requires `PyTorch>=2.0` to function properly with Taichi's `CUDA` arch.",
                    install=False,
                    extra=f"Check `https://pytorch.org/get-started/locally/`"
                          f" for more detailed guide on installing CUDA version of torch,"
                          f" or try using `{BCESolver.__name__}` under Taichi's `CPU` arch."
                )

                from .bce_ops.numpy_ops import NumpyOps as Ops
        else:
            from .bce_ops.numpy_ops import NumpyOps as Ops

        self._ops = Ops(self.kernel_dtype, self.device, self.bccb_shape, self.f_bccb_shape)

    def build_kernel(self, model):
        _init_table(
            *self.Tensor6,
            self.shape_cells,
            self.base_cell_sizes,
            self.kernel_dt
        )

        kernel_matrix_forward_separated(
            self.receiver_locations,
            self.xn, self.yn, self.zn, model,
            *self.Tmat6,
            kernel_dtype=self.kernel_dt,
            compressed=True,
            apply_susc_model=self.is_kernel_built_with_model
        )

        ops = self.ops
        for i, t in enumerate(self.Tensor6):
            # 对称情况下默认使用第一行填充，
            # 其内容为 [t_0, t_{-1}, ..., t_{-n+1}, t_{-n+1}, ..., t_{-1}]
            # 反对称时其内容应为 [t_0, t_1, ..., t_{n-1}, t_{-n+1}, ..., t_{-1}]
            # 在退磁核矩阵中， t_k = -t_{-k}，因此对应维度上的前半部分元素需要取反
            # BCESolver.BlockSkewSymmetric[i] 指定了在每个维度上是否为反对称
            padded = ops.tmp
            _extend_kernel_into_bccb_3d(
                t, padded, self.base_shape,
                *BCESolver.BlockSkewSymmetric[i]
            )

            ops.rfftn(padded, out=t, real=True)

    def solve(self, magnetization, model):
        ops = self.ops
        is_kernel_built_with_model = self.is_kernel_built_with_model
        use_complete_mesh = self.use_complete_mesh

        n_dim = 3
        tol = 1e-6  # TODO: 添加参数支持用户控制求解容差
        maxiter = 5000

        if use_complete_mesh:
            mask = slice(None)
        else:
            mask = ops.from_array(self.active_cells_mask)

        model = ops.from_array(model).reshape(-1, 1)

        b = ops.from_array(magnetization)
        b_2d = b.reshape(-1, n_dim)

        x = ops.array_like(b)

        # Y_x, Y_y, Y_z，用于存储频率域临时结果
        f_y = [
            ops.array(self.f_bccb_shape, dtype=ops.complex_kernel_dtype)
            for _ in range(3)
        ]

        Ap = ops.array_like(b, dtype=self.kernel_dtype)

        r = ops.copy(b)
        r_2d = r.reshape(-1, n_dim)

        p_t = ops.array(np.r_[self.base_shape, n_dim], dtype=self.kernel_dtype)
        p_2d_all = p_t.reshape(-1, n_dim)

        p_2d_all[mask] = b_2d

        def extract_p_2d():
            return p_2d_all[mask]

        def update_x():
            # x = x + alpha * p
            ops.iadd(x, p, alpha=alpha)

        def update_r():
            # r = r - alpha * Ap
            ops.iadd(r, Ap, alpha=-alpha)

        def update_p():
            # p = r + beta * p
            p_2d_all[:] *= beta
            p_2d_all[mask] += r_2d

        def adjust_p():
            pass

        def adjust_Ap():
            pass

        if not is_kernel_built_with_model:
            if use_complete_mesh:
                def adjust_p():
                    # 当磁化率不均匀时，build_kernel构建的核矩阵为 T 而非 I - TK
                    # 令 p' = Kp 以求 Tp' = TKp
                    p_2d_all[mask] *= model

                def adjust_Ap():
                    # 还原p变量
                    # Ap = p - Tp' = p - TKp
                    p_2d_all[mask] /= model
                    Ap[:] *= -1
                    Ap[:] += p
            else:
                # 令 p' = Kp ，因此 p' 的初始值为 Kb
                p_2d_all[mask] *= model

                def adjust_Ap():
                    # 当磁化率不均匀时，build_kernel构建的核矩阵为 T 而非 I - TK
                    # 令 p' = Kp
                    # 此时有 Ap = p'/K - Tp'
                    p_2d[:] /= model
                    Ap[:] *= -1
                    Ap[:] += p

                def update_p():
                    # p = r + beta * p
                    # => (Kp) = Kr + beta * (Kp)
                    p_2d_all[:] *= beta
                    p_2d_all[mask] += r_2d * model

        initial_rTr = self._reduce(r)

        if self.progress:
            progress_bar = SolverProgress(tol, maxiter)
        else:
            progress_bar = None

        old_rTr = initial_rTr

        for i in range(maxiter):
            p_2d = extract_p_2d()
            p = ops.flatten(p_2d)

            adjust_p()

            self._matvec(p_t, Ap, f_y, mask)

            adjust_Ap()

            pAp = self._reduce(p, Ap)

            alpha = old_rTr / pAp
            update_x()
            update_r()

            new_rTr = self._reduce(r)
            residual = np.sqrt(new_rTr.item())

            if progress_bar is not None:
                progress_bar.sync(residual)

            if residual < tol:
                break

            beta = new_rTr / old_rTr
            update_p()

            old_rTr = new_rTr

        if progress_bar is not None:
            progress_bar.close()

        if self.using_torch:
            # 保证返回值为numpy数组
            x = x.cpu().numpy()

        return x

    @property
    def device(self):
        return cast_taichi_arch_to_torch(self.arch)

    def _matvec(self, x, y, tmp, mask):
        ops = self.ops
        n_dim = 3

        for t in tmp:
            t[:] = 0

        for i in range(n_dim):
            f_v = ops.rfftn(x[..., i], s=self.bccb_shape)
            for j in range(n_dim):
                tmp[j] += f_v * self.Tensor33[j][i]

        indexer = tuple(slice(0, size) for size in self.base_shape)
        for i in range(n_dim):
            y[i::n_dim] = ops.flatten(ops.irfftn(tmp[i], s=self.bccb_shape)[indexer])[mask]

    def _reduce(self, p, q=None):
        if q is None:
            q = p

        return p @ q


@ti_kernel
def _init_table(
        Txx: ti.types.ndarray(),
        Txy: ti.types.ndarray(),
        Txz: ti.types.ndarray(),
        Tyy: ti.types.ndarray(),
        Tyz: ti.types.ndarray(),
        Tzz: ti.types.ndarray(),
        shape_cells: ti.types.ndarray(),
        base_cell_sizes: ti.types.ndarray(),
        dtype: ti.template()
):
    nx, ny, nz = shape_cells[0], shape_cells[1], shape_cells[2]
    bx, by, bz = base_cell_sizes[0], base_cell_sizes[1], base_cell_sizes[2]

    for k, j, i in ti.ndrange(nz, ny, nx):
        Txx[k, j, i] = ti.cast(bx * (i - 0.5), dtype)
        Txy[k, j, i] = ti.cast(bx * (i + 0.5), dtype)
        Txz[k, j, i] = ti.cast(by * (j - 0.5), dtype)
        Tyy[k, j, i] = ti.cast(by * (j + 0.5), dtype)
        Tyz[k, j, i] = ti.cast(bz * (k - 0.5), dtype)
        Tzz[k, j, i] = ti.cast(bz * (k + 0.5), dtype)


def _extend_kernel_into_bccb_3d_vanilla(
        kernel,
        out,
        s,
        opposite_symmetric_z,
        opposite_symmetric_y,
        opposite_symmetric_x
):
    padded = pad_back_symmetric(kernel, out.shape - s, out=out, mat_shape=s)
    for j in np.where([opposite_symmetric_z, opposite_symmetric_y, opposite_symmetric_x])[0]:
        # 对称情况下默认使用第一行填充，
        # 其内容为 [t_0, t_{-1}, ..., t_{-n+1}, t_{-n+1}, ..., t_{-1}]
        # 反对称时其内容应为 [t_0, t_1, ..., t_{n-1}, t_{-n+1}, ..., t_{-1}]
        # 其中 t_k = -t_{-k}，因此对应维度上的前半部分元素需要取反
        half_indexer = [slice(None)] * 3
        half_indexer[j] = slice(1, s[j])
        padded[tuple(half_indexer)] *= -1


@ti_kernel
def _extend_kernel_into_bccb_3d(
        kernel: ti.types.ndarray(),
        out: ti.types.ndarray(),
        s: ti.types.ndarray(),
        opposite_symmetric_x: ti.types.int8,
        opposite_symmetric_y: ti.types.int8,
        opposite_symmetric_z: ti.types.int8
):
    nx, ny, nz = s[0], s[1], s[2]
    ex, ey, ez = nx * 2 - 1, ny * 2 - 1, nz * 2 - 1

    for i, j, k in ti.ndrange(nx, ny, nz):
        val = kernel[i, j, k]
        out[i, j, k] = val  # 由于取反维度只能是0或2，因此负负得正，原始区域内的元素一定不需要取反

        if i >= 1:
            if opposite_symmetric_y ^ opposite_symmetric_z:
                out[ex - i, j, k] = -val
            else:
                out[ex - i, j, k] = val

        if j >= 1:
            if opposite_symmetric_z ^ opposite_symmetric_x:
                out[i, ey - j, k] = -val
            else:
                out[i, ey - j, k] = val

        if k >= 1:
            if opposite_symmetric_x ^ opposite_symmetric_y:
                out[i, j, ez - k] = -val
            else:
                out[i, j, ez - k] = val

        if i >= 1 and j >= 1:
            if opposite_symmetric_z:
                out[ex - i, ey - j, k] = -val
            else:
                out[ex - i, ey - j, k] = val

        if j >= 1 and k >= 1:
            if opposite_symmetric_x:
                out[i, ey - j, ez - k] = -val
            else:
                out[i, ey - j, ez - k] = val

        if k >= 1 and i >= 1:
            if opposite_symmetric_y:
                out[ex - i, j, ez - k] = -val
            else:
                out[ex - i, j, ez - k] = val

        if k >= 1 and j >= 1 and i >= 1:
            out[ex - i, ey - j, ez - k] = val


def pad_back_symmetric(mat, pad_width, *, out=None, mat_shape=None):
    n_dim = len(mat.shape)

    if mat_shape is not None:
        base_shape = mat_shape
    else:
        base_shape = np.asarray(mat.shape)

    if out is None:
        out = np.empty(base_shape + pad_width, dtype=mat.dtype)

    if mat is out:
        if mat_shape is None:
            base_shape = base_shape - pad_width
    else:
        indexer = tuple(slice(s) for s in base_shape)
        out[indexer] = mat[indexer]

    for i in range(len(pad_width)):
        if pad_width[i] == 0:
            continue

        indexer_dst = [slice(None)] * n_dim
        indexer_dst[i] = slice(-pad_width[i], None)

        indexer_src = [slice(None)] * n_dim
        indexer_src[i] = slice(base_shape[i] - 1, base_shape[i] - pad_width[i] - 1, -1)

        try:
            out[tuple(indexer_dst)] = out[tuple(indexer_src)]
        except ValueError:
            # 应该是torch的不支持负步长问题 "step must be greater than zero"
            # 采用手动逐元素拷贝
            # TODO: 优化torch下的padding效率（通过taichi kernel实现？）
            for j in range(base_shape[i] - pad_width[i], base_shape[i]):
                indexer_dst = [slice(None)] * n_dim
                indexer_dst[i] = -j

                indexer_src = [slice(None)] * n_dim
                indexer_src[i] = j

                out[tuple(indexer_dst)] = out[tuple(indexer_src)]

    return out


def cast_numpy_dtype_to_torch(dtype):
    import torch

    if type(dtype).__module__.startswith(torch.__name__):
        return dtype

    numpy_to_torch_dtype_dict = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        # np.uint16: torch.uint16,  # torch 2.0似乎暂时不包含无符号整形
        # np.uint32: torch.uint32,
        # np.uint64: torch.uint64,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128
    }

    for k, v in numpy_to_torch_dtype_dict.items():
        if k == dtype:
            return v
    else:
        warnings.warn(f'Unsupported numpy dtype {dtype} for torch, using float32 instead.')
        return torch.float32


def cast_taichi_arch_to_torch(arch):
    return arch.name


def cast_float_to_complex(dtype):
    numpy_to_torch_dtype_dict = {
        np.float64: np.complex64,
        np.complex64: np.complex64,
        np.complex128: np.complex128
    }

    if np.issubdtype(dtype, np.complexfloating):
        return dtype
    else:
        for k, v in numpy_to_torch_dtype_dict.items():
            if k == dtype:
                return v
        else:
            return np.complex64
