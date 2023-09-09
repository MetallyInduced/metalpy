import numpy as np
import tqdm


class QuickUnion:
    def __init__(self, n):
        """快速并查集
        """
        # TODO: 考察是否有必要使用int64作为索引类型
        self.unions = np.arange(n)

    def connect(self, a, b):
        rdst = self.find_root(a)
        rsrc = self.find_root(b)
        self.unions[rsrc] = rdst

    def find_root(self, a):
        root = self.unions[a]
        while root != a:
            a = root
            root = self.unions[a]

        return root

    def collapse(self, verbose=False):
        if verbose:
            span = tqdm.trange(len(self.unions), desc='Collapsing unions')
        else:
            span = range(len(self.unions))

        for i in span:
            a = i
            root = self.unions[a]
            while root != a:
                a = root
                root = self.unions[a]
            self.unions[i] = root
