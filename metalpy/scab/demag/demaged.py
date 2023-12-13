from __future__ import annotations

from typing import Union

from metalpy.mexin import Patch
from metalpy.scab import Tied
from metalpy.scab.distributed.policies import Distributable
from metalpy.utils.type import get_params_dict


class Demaged(Patch, Distributable):
    Priority = Tied.Priority + 1  # 保证在Tied后面注入mixin

    def __init__(
            self,
            factor=None,
            method=None,
            compressed_size: Union[int, float, None] = None,
            deterministic: Union[bool, str] = True,
            quantized: bool | None = None,
            symmetric: bool | None = None,
            progress: bool = False,
            kernel_dtype=None,
            **kwargs
    ):
        """为 `Simulation3DIntegral` 添加退磁效应计算

        Parameters
        ----------
        factor
            三轴退磁系数，如果为None，则采用数值方法求解
        method
            采用的数值求解算法，参考 `Demagnetization.methods`。默认为None，自动选择。
            可选选项：
                - Demagnetization.Compressed
                - Demagnetization.Seperated
                - Demagnetization.Integrated
        compressed_size
            压缩表尺寸，只针对`Demagnetization.Compressed`有效，用于指定压缩后的核矩阵哈希表最大大小，应尽可能小以满足内存要求。
            浮点数代表按完整核矩阵大小的比例设置，整数代表直接设置尺寸。
            如果大小不足或过大内部会给出提示信息
        deterministic
            采用确定性模式并行哈希，默认为True，牺牲一定计算效率和空间确保核矩阵正确。
            指定为CompressedForward.Optimal时直接采用unique函数计算理想压缩结果
        quantized
            采用量化模式压缩核矩阵。
            True则启用压缩，使用更小位宽来存储核矩阵，减少一定的空间需求，略微牺牲计算效率。
            （可能会报要求尺寸为2的幂的性能警告）。
            在CPU模式下会对性能造成较大影响，因此默认不启用。
            在GPU模式下性能影响较小，因此默认启用
        symmetric
            指示是否启用对称模式（当网格为规则网格时可以启用）。
            True则在检测到网格符合条件时启用对称模式，以对称矩阵形式存储核矩阵，减少一定的空间需求。
            False则禁用对称模式，无论是否符合条件。
            在CPU模式下会对性能造成较大影响，因此默认不启用。
            在GPU模式下性能影响较小，因此默认启用
        progress
            是否输出求解进度条，默认为False不输出
        kernel_dtype
            核矩阵数据类型，默认为None，自动从输入数据推断
        """
        super().__init__()
        self.params = get_params_dict(
            factor=factor,
            method=method,
            compressed_size=compressed_size,
            deterministic=deterministic,
            quantized=quantized,
            symmetric=symmetric,
            progress=progress,
            kernel_dtype=kernel_dtype,
            **kwargs
        )

    def apply(self):
        from SimPEG.potential_fields.magnetics import Simulation3DIntegral
        from metalpy.scab.demag.demaged_mixin import DemagedMixin

        self.add_mixin(Simulation3DIntegral, DemagedMixin, **self.params)

    def bind_context(self, ctx):
        super().bind_context(ctx)
