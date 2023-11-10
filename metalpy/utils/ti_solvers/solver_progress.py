from __future__ import annotations

from math import log10, inf

import tqdm

from metalpy.utils.numeric import limit_significand


class SolverProgress:
    def __init__(self, tol, maxiter, *, divergence_tol=27, mapper=log10, unit='logRES'):
        """通过残差来计算进度

        默认采用残差对数，不难注意到一些优化算法下，残差的对数呈线性下降趋势

        Parameters
        ----------
        tol
            目标残差
        maxiter
            最大运行代数
        divergence_tol
            发散检测阈值，当残差大于初始残差次数超过 `divergence_tol` 时，改为按迭代代数统计进度
        mapper
            残差转换函数，默认为限制有效位数下的对数函数
        unit
            残差单位，如果修改了 `mapper` 应该也要改一下单位
        """
        self.mapper = mapper
        self.unit = unit

        self.postfixes: dict[str, float | str] = {
            'iter': 0,
            'residual': inf,
            'target': tol
        }
        self._n_iter = 0
        self.log_res = 0
        self.progress_bar = None
        self.progress_by_res_counter = divergence_tol

        self.end = self._update_residual(tol)
        self.origin = 0
        self.maxiter = maxiter

    def sync(self, residual):
        self.n_iter += 1
        log_res = self._update_residual(residual)

        if self.progress_bar is None:
            # 先等待残差下降才启动进度条
            self.origin = origin = log_res
            self.progress_bar = tqdm.tqdm(
                total=origin - self.end,
                unit=self.unit,
                postfix=self.postfixes
            )
        elif self.progress_by_res_counter >= 0:
            total = self.origin - log_res
            if total < 0:
                if self.progress_by_res_counter > 0:
                    total = 0
                self.progress_by_res_counter -= 1
            if total >= 0:
                if total > self.total:
                    total = self.total
                delta = total - self.progress_bar.n
                self.progress_bar.set_postfix(self.postfixes, refresh=False)
                self.progress_bar.update(delta)
            else:
                # 就要发散就要发散，给您改为基于迭代代数的进度条
                self.postfixes.pop('iter')  # 不需要再在后缀中展示当前代数
                self.progress_bar.unit = 'it'
                self.progress_bar.total = self.maxiter
                self.progress_bar.last_print_n = self.n_iter
                self.progress_bar.n = self.n_iter + 1
                self.progress_bar.set_postfix(self.postfixes, refresh=False)
                self.progress_bar.refresh()
        else:
            self.progress_bar.set_postfix(residual=self.res_str, refresh=False)
            self.progress_bar.update(1)

    def close(self):
        if self.progress_bar is not None:
            self.progress_bar.close()

    @property
    def n_iter(self):
        return self._n_iter

    @n_iter.setter
    def n_iter(self, v):
        self._n_iter = v
        if 'iter' in self.postfixes:
            self.postfixes['iter'] = v

    @property
    def res_str(self):
        return self.postfixes['residual']

    @res_str.setter
    def res_str(self, v):
        self.postfixes['residual'] = v

    @property
    def total(self):
        return self.progress_bar.total

    def _update_residual(self, residual):
        log_res = limit_significand(self.mapper(residual), bits=6)
        res_str = f'{residual:.2e}'

        self.res_str = res_str
        self.log_res = log_res

        return log_res
