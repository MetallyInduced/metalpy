from enum import Enum
from typing import Union, Callable


class MixMode(Enum):
    @staticmethod
    def dispatch(mode):
        if isinstance(mode, MixMode):
            name = str(mode.value)
            return getattr(MixMode, name)
        elif callable(mode):
            return mode
        else:
            raise ValueError(f'Mix mode must be either MixMode or function, got {type(mode)} instead.')

    @staticmethod
    def override(prev_layer, current_layer):
        return current_layer

    @staticmethod
    def keep_original(prev_layer, current_layer):
        return prev_layer

    @staticmethod
    def max(prev_layer, current_layer):
        mask = prev_layer < current_layer
        prev_layer[mask] = current_layer[mask]
        return prev_layer

    @staticmethod
    def min(prev_layer, current_layer):
        mask = prev_layer > current_layer
        prev_layer[mask] = current_layer[mask]
        return prev_layer

    @staticmethod
    def farther_from_zero(prev_layer, current_layer):
        mask = abs(prev_layer) < abs(current_layer)
        prev_layer[mask] = current_layer[mask]
        return prev_layer

    @staticmethod
    def closer_to_zero(prev_layer, current_layer):
        mask = abs(prev_layer) > abs(current_layer)
        prev_layer[mask] = current_layer[mask]
        return prev_layer

    @staticmethod
    def mean(prev_layer, current_layer):
        return (prev_layer + current_layer) / 2

    Override = override.__func__.__name__  # 直接覆盖
    KeepOriginal = keep_original.__func__.__name__  # 保留原值
    Max = max.__func__.__name__  # 取最大值 / 正数意义下的并集 / 负数意义下的交集
    Min = min.__func__.__name__  # 取最小值 / 正数意义下的交集 / 负数意义下的并集
    AbsMax = FartherFromZero = farther_from_zero.__func__.__name__  # 取绝对值意义下最大值 / 更远离原点的值 / 同符号意义下的并集
    AbsMin = CloserToZero = closer_to_zero.__func__.__name__  # 取绝对值意义下最小值 / 更接近原点的值 / 同符号意义下的交集
    Mean = mean.__func__.__name__  # 取平均值


Mixer = Union[MixMode, Callable]


def dhashable_mixer(mixer: Mixer):
    """返回mixer的哈希值

    Parameters
    ----------
    mixer
        混合器名或函数
    """
    if isinstance(mixer, MixMode):
        return mixer.value
    else:
        import cloudpickle
        return cloudpickle.dumps(mixer)
