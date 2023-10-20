from typing import Iterable

import numpy as np

from metalpy.scab.utils.misc import Field
from .utils import get_demag_apparent_k


class FactoredDemagnetization:
    def __init__(self, n, source_field=None):
        """
        依据退磁因子，计算退磁作用下的磁化强度，通过退磁因子计算视磁化率

        Parameters
        ----------
        n
            三轴退磁系数
        source_field
            外部场源，也可以在dpred时指定，主要为保持和Simulation的一致性

        Notes
        -----
        当某一方向系数为None时自动推断，所有None值假设为相同的值，且值为 (1 - sum(n[n != None])) / len(n[n == None])
        """
        super().__init__()
        if isinstance(n, Iterable):
            n = np.asarray(n)
            if np.any(n == None):
                n[n == None] = (1 - sum(n[n != None])) / len(n[n == None])
                n = n.astype(np.float64)
        self.N = n
        self.source_field = source_field

    def dpred(self, model, source_field: Field = None):
        """
        Parameters
        ----------
        model
            array-like(nC,)，磁化率模型
        source_field
            外部场源，若为空，则返回视磁化率矩阵，否则返回三轴等效磁化率

        Returns
        -------
        ret
            array(nC, 3)，
            若source_field为空，返回三轴视磁化率；
            若source_field非空，返回在该场作用下的三轴等效磁化率（三轴磁化强度除以场源强度）
        """
        if source_field is None:
            source_field = self.source_field

        model = np.asarray(model)
        k = get_demag_apparent_k(model, self.N)

        if source_field is not None:
            k = k * Field(source_field).unit_vector

        return k
