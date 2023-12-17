from math import sqrt

import taichi as ti
from taichi.lang.util import to_taichi_type

from metalpy.utils.taichi import copy_from
from metalpy.utils.ti_solvers.solver_progress import SolverProgress


def cg(
        A: ti.linalg.LinearOperator,
        b,
        x0=None,
        tol=1e-6,
        maxiter=5000,
        progress=False
):
    """基于线性算符的共轭梯度法求解器

    基于ti.linalg.taichi_cg_solver修改而来，实现进度条，并处理精度丢失警告信息

    Parameters
    ----------
    A
        线性算符，通过LinearOperator包装
    b
        常数项
    x0
        初始解，默认为None，自动初始化为0
    tol
        收敛容差
    maxiter
        最大迭代代数
    progress
        是否启用进度条

    Returns
    -------
    x
        求解结果，以 `ti.field` 形式返回，调用方可以根据需要转换为numpy、torch或其它数组形式

    Notes
    -----
    注意到正常收敛情形下， `残差的对数` 与 `迭代代数` 呈线性关系，可以作为进度条依据。

    如果残差出现波动无法正常减少，则改为基于迭代次数的进度条，以 `maxiter` 为总代数。
    """
    solver_dtype = to_taichi_type(b.dtype)
    size = b.shape

    x = ti.field(dtype=solver_dtype, shape=size)

    vector_fields_builder = ti.FieldsBuilder()
    b, _b = ti.field(dtype=solver_dtype), b
    p = ti.field(dtype=solver_dtype)
    r = ti.field(dtype=solver_dtype)
    Ap = ti.field(dtype=solver_dtype)
    vector_fields_builder.dense(ti.i, size).place(b)
    vector_fields_builder.dense(ti.i, size).place(p, r, Ap)
    vector_fields_snode_tree = vector_fields_builder.finalize()

    scalar_builder = ti.FieldsBuilder()
    alpha = ti.field(dtype=solver_dtype)
    beta = ti.field(dtype=solver_dtype)
    scalar_builder.place(alpha, beta)
    scalar_snode_tree = scalar_builder.finalize()

    copy_from(b, _b)
    has_initial = x0 is not None

    @ti.kernel
    def _init():
        if ti.static(has_initial):
            for I in ti.grouped(x):
                p[I] = r[I] = b[I] - Ap[I]
        else:
            for I in ti.grouped(x):
                p[I] = r[I] = b[I]

    def init():
        if has_initial:
            copy_from(x, x0)
            A._matvec(x, Ap)  # compute Ap = A x p
        _init()

    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for I in ti.grouped(p):
            result += p[I] * q[I]
        return result

    @ti.kernel
    def update_x():
        for I in ti.grouped(x):
            x[I] += alpha[None] * p[I]

    @ti.kernel
    def update_r():
        for I in ti.grouped(r):
            r[I] -= alpha[None] * Ap[I]

    @ti.kernel
    def update_p():
        for I in ti.grouped(p):
            p[I] = r[I] + beta[None] * p[I]

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

        old_rTr = initial_rTr

        # -- Main loop --
        for i in range(maxiter):
            A._matvec(p, Ap)  # compute Ap = A x p
            pAp = reduce(p, Ap)
            update(alpha, old_rTr / pAp)
            update_x()
            update_r()
            new_rTr = reduce(r, r)
            residual = sqrt(new_rTr)

            if progress_bar is not None:
                progress_bar.sync(residual)

            if residual < tol:
                break

            update(beta, new_rTr / old_rTr)
            update_p()
            old_rTr = new_rTr

        if progress_bar is not None:
            progress_bar.close()

    solve()

    vector_fields_snode_tree.destroy()
    scalar_snode_tree.destroy()

    return x
