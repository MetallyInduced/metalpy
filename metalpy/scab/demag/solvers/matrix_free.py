from math import sqrt

import numpy as np
import taichi as ti
import tqdm
from taichi import TaichiTypeError, TaichiRuntimeError


def conjugate_gradient(
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
    Ap = ti.field(dtype=solver_dtype)
    vector_fields_builder.dense(ti.ij, size).place(p, r, Ap)
    vector_fields_snode_tree = vector_fields_builder.finalize()

    scalar_builder = ti.FieldsBuilder()
    alpha = ti.field(dtype=solver_dtype)
    beta = ti.field(dtype=solver_dtype)
    scalar_builder.place(alpha, beta)
    scalar_snode_tree = scalar_builder.finalize()

    @ti.kernel
    def init():
        for I in ti.grouped(x):
            r[I] = b[I]
            p[I] = 0.0
            Ap[I] = 0.0

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

    def truncated_log(residual):
        return round(np.log10(residual), 3)

    def solve():
        init()
        initial_rTr = reduce(r, r)

        progress_bar = None
        progress_by_res_counter = 3  # 发散计数器，残差大于初始残差时减去1，低于0后视为发散，转为按迭代代数统计进度
        origin = initial_rTr
        postfixes = {
            'iter': 0,
            'residual': np.inf,
            'target': tol
        }

        old_rTr = initial_rTr
        update_p()
        # -- Main loop --
        for i in range(maxiter):
            A._matvec(p, Ap)  # compute Ap = A x p
            pAp = reduce(p, Ap)
            alpha[None] = old_rTr / pAp
            update_x()
            update_r()
            new_rTr = reduce(r, r)
            if sqrt(new_rTr) < tol:
                break
            beta[None] = new_rTr / old_rTr
            update_p()
            old_rTr = new_rTr

            if progress:
                res = sqrt(new_rTr)
                log_res = truncated_log(res)
                res_str = f'{res:.2e}'
                postfixes['residual'] = res_str

                if progress_bar is None:
                    # 先等待残差下降才启动进度条
                    origin = log_res
                    end = truncated_log(tol)
                    postfixes['iter'] = 0
                    progress_bar = tqdm.tqdm(
                        total=origin - end,
                        unit='logRES',
                        postfix=postfixes
                    )
                elif progress_by_res_counter >= 0:
                    n = origin - log_res
                    if n < 0:
                        if progress_by_res_counter > 0:
                            n = 0
                        progress_by_res_counter -= 1

                    if n >= 0:
                        postfixes['iter'] = i + 1
                        progress_bar.set_postfix(postfixes, refresh=False)
                        progress_bar.update(n - progress_bar.n)
                    else:
                        # 发得一手好散，改为基于代数的进度条
                        postfixes.pop('iter')  # 不需要再在后缀中展示当前代数
                        progress_bar.unit = 'it'
                        progress_bar.total = maxiter
                        progress_bar.last_print_n = i
                        progress_bar.n = i + 1
                        progress_bar.set_postfix(postfixes, refresh=False)
                        progress_bar.refresh()
                else:
                    progress_bar.set_postfix(residual=res_str, refresh=False)
                    progress_bar.update(1)

        if progress_bar is not None:
            res = sqrt(new_rTr)
            log_res = truncated_log(res)
            res_str = f'{res:.2e}'
            postfixes['residual'] = res_str

            if progress_by_res_counter >= 0:
                # 重设进度条最大值，保证进度条最终会跑完
                n = origin - log_res
                progress_bar.total = n
                postfixes['iter'] = i + 1
                progress_bar.set_postfix(postfixes, refresh=False)
                progress_bar.update(n - progress_bar.n)
            else:
                progress_bar.set_postfix(residual=res_str, refresh=False)
                progress_bar.update(1)

            progress_bar.close()

    solve()
    vector_fields_snode_tree.destroy()
    scalar_snode_tree.destroy()
