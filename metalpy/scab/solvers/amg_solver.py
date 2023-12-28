import json
import re
import warnings

import numpy as np
from pymatsolver.solvers import Base

from metalpy.utils.file import make_cache_file
from metalpy.utils.ti_solvers.solver_progress import ProgressList, SolverProgress


class AMGSolver(Base):
    def __init__(self, A, cuda=False, rtol=1e-6, progress=False):
        """基于AMG的稀疏线性方程组求解器

        Parameters
        ----------
        A
            稀疏系数矩阵
        cuda
            是否使用CUDA加速（基于 `pyamgx` ），否则使用 `pyamg`
        rtol
            相对残差
        progress
            是否显示求解进度
        """
        super().__init__(A)

        self.cuda = cuda

        self.rtol = rtol
        self.progress = progress

    def _solve1(self, rhs):
        if self.cuda:
            try:
                return self._solve_amgx(rhs)
            except ImportError:
                warnings.warn(
                    f'Failed to import `pyamgx`, ignoring `cuda` option.'
                    f' Falling back to `pyamg`.'
                )

        return self._solve_amg(rhs)

    def _solveM(self, rhs):
        return np.vstack([self._solve1(r)[:, np.newaxis] for r in rhs.T])

    def _solve_amg(self, rhs):
        import pyamg

        rhs = rhs.squeeze()
        norm_b = np.linalg.norm(rhs)
        rtol = self.rtol
        atol = rtol * norm_b
        max_iter = np.iinfo(np.int32).max

        if self.progress:
            progress = ProgressList(atol)
        else:
            progress = None

        solver = pyamg.ruge_stuben_solver(self.A)
        x, _ = solver.solve(
            b=rhs,
            tol=rtol,
            residuals=progress,
            maxiter=max_iter,
            return_info=True
        )

        return x

    def _solve_amgx(self, rhs):
        import pyamgx

        rhs = rhs.squeeze()
        norm_b = np.linalg.norm(rhs)
        rtol = self.rtol
        atol = rtol * norm_b
        max_iter = np.iinfo(np.int32).max

        pyamgx.initialize()

        if self.progress:
            pyamgx.register_print_callback(_ResidualLogger(atol))

        cfg = pyamgx.Config()

        params = {
            "config_version": 2,
            "solver": {
                "print_grid_stats": 1,
                "solver": "AMG",
                "print_solve_stats": 1,
                "interpolator": "D2",
                "presweeps": 1,
                "obtain_timings": 1,
                "max_iters": max_iter,
                "monitor_residual": 1,
                "convergence": "ABSOLUTE",
                "scope": "main",
                "max_levels": 30,
                "cycle": "CG",
                "tolerance": atol,
                "norm": "L2",
                "postsweeps": 1
            }
        }
        with open(make_cache_file('amgx.json'), 'w') as fp:
            json.dump(params, fp)
        cfg.create_from_file(fp.name.encode())

        resources = pyamgx.Resources().create_simple(cfg)

        mat = pyamgx.Matrix().create(resources)
        b = pyamgx.Vector().create(resources)
        x = pyamgx.Vector().create(resources)

        mat.upload_CSR(self.A)
        b.upload(rhs)
        x.upload(rhs)

        solver = pyamgx.Solver()
        solver.create(resources, cfg)
        solver.setup(mat)

        solver.solve(b, x, zero_initial_guess=True)

        ans = x.download()

        solver.destroy()
        x.destroy()
        b.destroy()
        mat.destroy()
        resources.destroy()
        cfg.destroy()
        pyamgx.finalize()

        return ans


class _ResidualLogger:
    def __init__(self, tol, maxiter=None):
        self.progress = SolverProgress(tol, maxiter)
        self.running = False

    def __call__(self, msg):
        if 'residual' in msg:
            self.running = True
            msg = re.split(r'-+', msg, maxsplit=1)[1]

        if 'Final Residual' in msg:
            self.running = False

        if self.running:
            res = float(re.split(r'\s+', msg.strip(), maxsplit=4)[2])
            self.progress.sync(res)
