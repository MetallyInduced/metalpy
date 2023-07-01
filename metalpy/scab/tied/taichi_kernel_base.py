from __future__ import annotations

from enum import IntFlag
import functools

from metalpy.mexin import Mixin


class Profiler(IntFlag):
    Disabled = 0     # 无Profiler
    Scoped = 1       # 输出Taichi处理编译等中间过程的耗时情况
    CountKernel = 2  # 输出Taichi Kernel的运行情况，但相同kernel合并
    TraceKernel = 4  # 输出所有Taichi Kernel的运行情况

    Default = Scoped | CountKernel

    @property
    def needs_kernel_profiler(self):
        return self > 1

    @staticmethod
    def of(val):
        if val is True:
            return Profiler.Default
        elif val is False:
            return Profiler.Disabled
        else:
            return val


class TiedMixin(Mixin):
    def __init__(self, this, profile: Profiler | bool):
        super().__init__(this)

        self.profile = Profiler.of(profile)


def tied_profile(fn):
    """用于包装需要对TaichiKernel进行profile的成员函数，成员函数所在类需要包含profile属性
    （继承自TaichiKernelMixin自动包含该属性）

    kernel_profiler需要在ti.init时指定`kernel_profiler=True`
    """
    @functools.wraps(fn)
    def wrapper(self: TiedMixin, *args, **kwargs):
        import taichi as ti

        ret = fn(self, *args, **kwargs)
        profile = self.profile

        if profile & Profiler.Scoped:
            ti.profiler.print_scoped_profiler_info()

        if profile & Profiler.CountKernel:
            ti.profiler.print_kernel_profiler_info('count')

        if profile & Profiler.TraceKernel:
            ti.profiler.print_kernel_profiler_info('trace')

        if profile.needs_kernel_profiler:
            ti.profiler.clear_kernel_profiler_info()

        return ret

    return wrapper
