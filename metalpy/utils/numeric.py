import numpy as np


def limit_significand(val, bits=None, tol=None):
    """将 val 转换为 IEEE754 格式浮点数可以准确表示的形式

    可选截断模式包含：

    - 按二进制有效位数截断
    - 按与原值的差距进行阶段

    Parameters
    ----------
    val
        需要限制有效位数的浮点数
    bits
        保留的二进制有效位数
    tol
        与原值的容差

    Returns
    -------
    truncated_val
        截断后的值
    """
    val = np.asarray(val)
    dt = val.dtype
    if np.issubdtype(dt, np.integer):
        return val

    finfo = np.finfo(dt)

    if bits is None and tol is None:
        tol = 10 ** -(finfo.precision // 2)

    if bits is None:
        bits = finfo.bits

    if tol is None:
        tol = -1

    int_part = val.astype(np.int64)
    float_part = val - int_part
    fp = 0
    a = 1
    for i in range(bits):
        sign = np.sign(float_part - fp)
        a /= 2
        if a <= tol:
            break
        else:
            fp += sign * a

    return int_part + fp

