import numpy as np

from metalpy.mepa import SingleTaskAllocator


def test_allocator_completeness():
    """测试分配器能否在存在浮点权重时保证按每个权重slice一次后，不会因舍入误差导致数据剩余
    """
    weights = [0.91, 0.4826, 0.11, 0.6739, 0.759, 0.0001]
    alloc = SingleTaskAllocator(np.arange(100), total=sum(weights))
    for w in weights:
        alloc.slice(w)

    assert alloc.finished
