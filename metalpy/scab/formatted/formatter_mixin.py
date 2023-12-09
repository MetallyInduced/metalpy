from __future__ import annotations

from metalpy.mexin import Mixin, mixin


class FormatterMixin(Mixin):
    def __init__(self, this, pandas=False, locations=False):
        """所有Formatter插件的基类，子类通过实现 `format_data` 以对data进行格式化操作

        Parameters
        ----------
        pandas
            如果为True，则格式化为 `pandas.DataFrame` ，行名为数据名（例如 'tmi' ， 'bx' 等）
        locations
            如果为True，则在格式化结果的左侧附加观测点的坐标列

        Notes
        -----
        如果存在多组数据（多个receiver组，其数值结果语义上无法直接合并在一个矩阵中），则视情况讨论：
            1. 如果 `pandas=True` ，则分别构造pandas矩阵，然后纵向拼接（缺失的分量会以nan形式存在）
            2. 如果 `pandas=False` ，则分别构造numpy矩阵，然后以多个numpy矩阵的list形式返回

        子类可能还需要根据对应正演类的预测例程，对对应的预测函数进行覆盖
        （例如针对Simulation3DDifferential需要额外在mixin中覆盖 `projectFields` ）
        """
        super().__init__(this)
        self.pandas = pandas
        self.locations = locations

    def format_data(self, this, data):
        raise NotImplementedError()

    @mixin.after(keep_retval='ret')
    def dpred(self, *_, ret, **__):
        return self.format_data(ret)
