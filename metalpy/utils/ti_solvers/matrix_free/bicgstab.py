from math import sqrt

import taichi as ti
from taichi import TaichiTypeError, TaichiRuntimeError

from ..solver_progress import SolverProgress


def bicgstab(
        A: ti.linalg.LinearOperator,
        b,
        x,
        tol=1e-6,
        maxiter=5000,
        progress=False
):
    """基于线性算符的线性求解器

    基于ti.linalg.taichi_cg_solver修改而来，实现进度条，并处理精度丢失警告信息

    Parameters
    ----------
    A
        线性算符，通过LinearOperator包装
    b
        常数项
    x
        未知量，同时用于存储求解结果
    tol
        收敛容差
    maxiter
        最大迭代代数
    progress
        是否启用进度条

    Notes
    -----
    注意到正常收敛情形下， `残差的对数` 与 `迭代代数` 呈线性关系，可以作为进度条依据。

    如果残差出现波动无法正常减少，则改为基于迭代次数的进度条，以 `maxiter` 为总代数。
    """
    if b.dtype != x.dtype:
        raise TaichiTypeError(f"Dtype mismatch b.dtype({b.dtype}) != x.dtype({x.dtype}).")
    if str(b.dtype) == "f32":
        solver_dtype = ti.f32
    elif str(b.dtype) == "f64":
        solver_dtype = ti.f64
    else:
        raise TaichiTypeError(f"Not supported dtype: {b.dtype}")
    if b.shape != x.shape:
        raise TaichiRuntimeError(f"Dimension mismatch b.shape{b.shape} != x.shape{x.shape}.")

    size = b.shape
    vector_fields_builder = ti.FieldsBuilder()
    p = ti.field(dtype=solver_dtype)
    r = ti.field(dtype=solver_dtype)
    rstar = ti.field(dtype=solver_dtype)
    Ap = ti.field(dtype=solver_dtype)
    s = ti.field(dtype=solver_dtype)
    As = ti.field(dtype=solver_dtype)

    vector_fields_builder.dense(ti.ij, size).place(p, r, Ap, rstar, s, As)
    vector_fields_snode_tree = vector_fields_builder.finalize()

    scalar_builder = ti.FieldsBuilder()
    alpha = ti.field(dtype=solver_dtype)
    beta = ti.field(dtype=solver_dtype)
    omega = ti.field(dtype=solver_dtype)
    scalar_builder.place(alpha, beta, omega)
    scalar_snode_tree = scalar_builder.finalize()

    @ti.kernel
    def init():
        for I in ti.grouped(x):
            r[I] = b[I]
            rstar[I] = r[I]
            p[I] = r[I]
            Ap[I] = 0.0

    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for I in ti.grouped(p):
            result += p[I] * q[I]
        return result

    @ti.kernel
    def copy(dst: ti.template(), src: ti.template()):
        for I in ti.grouped(p):
            dst[I] += src[I]

    @ti.kernel
    def update_x():
        for I in ti.grouped(x):
            x[I] = x[I] + alpha[None] * p[I] + omega[None] * s[I]

    @ti.kernel
    def update_r():
        for I in ti.grouped(r):
            r[I] = s[I] - omega[None] * As[I]

    @ti.kernel
    def update_p():
        for I in ti.grouped(p):
            p[I] = r[I] + beta[None] * (p[I] - omega[None] * Ap[I])  # p = r + beta * (p - omega * AMp)

    @ti.kernel
    def update_s():
        for I in ti.grouped(p):
            s[I] = r[I] - alpha[None] * Ap[I]

    @ti.kernel
    def update(dst: ti.template(), val: solver_dtype):
        # 防止在1.7.0下直接python scope执行赋值
        # dst[None] = val  # 警告信息：Assign may lose precision: unknown <- f64
        dst[None] = val

    def solve():
        init()
        initial_rTr = reduce(r, r)

        if progress:
            progress_bar = SolverProgress(tol, maxiter)
        else:
            progress_bar = None

        old_rrstar = initial_rTr * initial_rTr

        # update_p()
        # -- Main loop --
        for i in range(maxiter):
            A._matvec(p, Ap)  # compute Ap = A x p
            update(alpha, old_rrstar / reduce(rstar, Ap))

            update_s()

            A._matvec(s, As)  # compute As = A x s
            update(omega, reduce(As, s) / reduce(As, As))  # np.inner(AMs.conjugate(), s)/np.inner(AMs.conjugate(), AMs)

            update_x()
            update_r()

            new_rrstar = reduce(rstar, r)
            update(beta, (new_rrstar / old_rrstar) * (alpha[None] / omega[None]))
            old_rrstar = new_rrstar

            update_p()

            rTr = reduce(r, r)
            residual = sqrt(rTr)

            if progress_bar is not None:
                progress_bar.sync(residual)

            if residual < tol:
                break

        if progress_bar is not None:
            progress_bar.close()

    solve()
    vector_fields_snode_tree.destroy()
    scalar_snode_tree.destroy()
