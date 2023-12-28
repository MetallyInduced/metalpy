import warnings

import numpy as np
import taichi as ti
from pymatsolver.solvers import Base
from taichi.lang.util import to_taichi_type

from metalpy.utils.taichi import ti_sparse_from, get_arch, ti_ndarray_from, ti_size_max
from metalpy.utils.type import notify_package


class TiedSparseSolver(Base):
    def __init__(self, A, fallback=None, fallback_opts=None, rtol=1e-6):
        """基于Taichi的稀疏线性方程组求解器

        Parameters
        ----------
        A
            稀疏系数矩阵
        fallback
            Taichi求解器不可用时的备用求解器
        fallback_opts
            Taichi求解器不可用时的备用求解器参数

        Notes
        -----
        Taichi （截至1.7.0） 内置的稀疏求解器在CPU上性能表现较差，故不采用
        """
        # GPU版本基于 `ti.linalg.SparseCG`
        # CPU版本会调用fallback作为求解器
        super().__init__(A)

        if fallback_opts is None:
            fallback_opts = {}

        self.fallback = fallback
        self.fallback_opts = fallback_opts

        self.rtol = rtol

    def _solve1(self, rhs):
        rhs = rhs.squeeze()
        norm_b = np.linalg.norm(rhs)
        rtol = self.rtol
        atol = rtol * norm_b

        if get_arch() == ti.cpu:
            warnings.warn(
                f'`TiedSparseSolver` for `ti.cpu` has only limited support.'
                f' Falling back to `{self.fallback.__name__}`.'
            )
            fallback = check_solver(self.fallback)
            return fallback(self.A, **self.fallback_opts) * rhs
        else:
            dtype = np.float32
            dt = to_taichi_type(dtype)

            solver = ti.linalg.SparseCG(
                ti_sparse_from(self.A, dtype=dt),
                ti_ndarray_from(rhs, dtype=dt),
                atol=atol,
                max_iter=ti_size_max
            )

            x, success = solver.solve()

            return x.to_numpy()

    def _solveM(self, rhs):
        return np.vstack([self._solve1(r)[:, np.newaxis] for r in rhs.T])


def check_solver(solver):
    if solver is None:
        try:
            from pymatsolver import Pardiso as DefaultSolver
        except ImportError:
            from SimPEG.utils.solver_utils import SolverLU as DefaultSolver
            notify_package(
                pkg_name='pydiso',
                reason=f'Consider using `Pardiso` for better performance'
                       f' (currently `{DefaultSolver.__name__}`).',
                install='conda install pydiso'
            )
        solver = DefaultSolver

    return solver
