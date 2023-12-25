# 对应 taichi.types.annotations
from .lang.util import TaichiType


class sparse_matrix_builder(TaichiType):
    def __getitem__(self, item): ...

    def __setitem__(self, key, value): ...

    def __to_taichi_type__(self):
        import taichi as ti
        return ti.types.sparse_matrix_builder()
