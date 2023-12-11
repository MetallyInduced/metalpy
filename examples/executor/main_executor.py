from time import sleep

import numpy as np

from metalpy.mepa import ThreadExecutor

if __name__ == '__main__':
    with ThreadExecutor(4) as executor:
        progress = executor.progress()

        def task(some_data):
            ret = []
            for d in progress.iters(some_data):
                ret.append(d + 1)
                # 1秒处理4个数据，4个线程因此处理速度应接近 16it/s
                # 此处由于通过sleep模拟负载，因此不需要考虑GIL，真实负载可能需要使用ProcessExecutor
                sleep(0.25)
            return np.asarray(ret)

        # 生成输入数据
        data = np.arange(100)

        # 构造任务分配器
        alloc = executor.arrange(data, shuffle=True)

        # 基于alloc自动分配任务到各个worker
        tasks = executor.distribute(task, alloc)

        # 分布式计算任务主要可以分为两类：
        #   map型运算：输入和输出长度相同，一一对应，例如 y = x + 1
        #   reduce型运算：输出长度小于输入长度，一般输出长度为1，例如 y = sum(x)
        # 对于map型运算，可以通过alloc.reassemble将结果重新组装为原始数据的顺序
        result = alloc.reassemble(executor.gather(tasks))

        assert np.array_equal(result, data + 1)
