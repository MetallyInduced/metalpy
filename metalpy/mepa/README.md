MEtalpy PArallel
===========================

MEPA is an abstraction layer for parallel execution,
providing unified APIs to submit tasks, gather tasks and communicate with workers 
through threads, [processes (loky)](https://github.com/joblib/loky) and [Dask.distributed](https://github.com/dask/distributed) workers.

Installation
------------
MEPA is now a submodule of metalpy, which can be installed using pip,
with extra dependencies required by MEPA:

```console
pip install "metalpy[mepa]"
```

Features
--------
### Submit and Gather Tasks

Submit and gather tasks just like native Python, in a more flexible manner.

```python
from metalpy.mepa import ProcessExecutor

def func_add(a, b):
    return a + b


def func_mul(a, b):
    return a * b


def func_div(a, b):
    return a / b


def main():
    with ProcessExecutor(3) as executor:
        futures = {
            'add': executor.submit(func_add, 2, 1),
            'mul': executor.submit(func_mul, 2, 1),
            'div': executor.submit(func_div, 2, 1),
        }
        print(executor.gather(futures))

if __name__ == '__main__':
    main()
```

### Unified Communication API

MEPA comes with a unified API for distributed communication.

```python
from time import sleep
import tqdm
from metalpy.mepa import ProcessExecutor


class SomeTask:
    def __init__(self, task_length):
        self.task_length = task_length

    def do(self, ctx):
        for i in range(self.task_length):
            sleep(0.5)
            ctx.fire(1)

def main():
    n_tasks = 2
    task_length = 10

    progress = tqdm.tqdm(total=n_tasks * task_length)
    
    with ProcessExecutor(4) as executor:
        task = SomeTask(task_length)  # tasks that takes some time
        ctx = executor.register(lambda t: progress.update(t))  # register communication context
        executor.gather([executor.submit(task.do, ctx) for _ in range(n_tasks)])

if __name__ == '__main__':
    main()
```

### Parallel Utility

MEPA simplifies task management in several aspects.

```python
import numpy as np
from time import sleep
from metalpy.mepa import ProcessExecutor


with ProcessExecutor(4) as executor:
    progress = executor.progress()

    def task(some_data):
        ret = []
        for d in progress.iters(some_data):
            ret.append(d + 1)
            sleep(1)
        return np.asarray(ret)

    data = np.arange(20)

    # shuffle and allocates data according to executor's workers
    alloc = executor.arrange(data, shuffle=True)

    # distribute data to all workers
    tasks = executor.distribute(task, alloc)

    # for output that corresponds to input element-wisely
    # gather and assemble results
    result = alloc.reassemble(executor.gather(tasks))

    assert np.array_equal(result, data + 1)
```

### Dask Wrapper

MEPA has wrapped `Dask.distributed` to automatically take care of script dependencies.

```python
# add.py
def func(a, b):
    return a + b


# main.py
from metalpy.mepa import DaskExecutor
from add import func


def main():
    # automatically uploads `add.py` to cluster
    with DaskExecutor('address to your cluster') as executor:
        executor.submit_and_gather(func, 1, 2)

if __name__ == '__main__':
    main()
```
