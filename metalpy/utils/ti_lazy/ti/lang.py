from abc import abstractmethod, ABC


def to_taichi_type(dtype):
    magic = getattr(dtype, '__to_taichi_type__', None)

    if magic is not None:
        if isinstance(dtype, type):
            return magic(dtype())
        else:
            return magic()

    from taichi.lang.util import to_taichi_type as _to_taichi_type
    return _to_taichi_type(dtype)


class TaichiType(ABC):
    mapping = {}

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def __to_taichi_type__(self):
        raise NotImplementedError()


class ti_template(TaichiType):
    def __init__(self, tensor=None, dim=None):
        super().__init__(tensor, dim)

    def __to_taichi_type__(self):
        import taichi as ti
        return ti.types.template(*self.args, **self.kwargs)

    @property
    def shape(self) -> tuple: return ()

    def __getitem__(self, item): ...

    def __setitem__(self, key, value): ...


class ti_ndarray(ti_template):
    def __init__(self,
                 dtype=None,
                 ndim=None,
                 element_dim=None,
                 element_shape=None,
                 field_dim=None,
                 needs_grad=None,
                 boundary="unsafe"):
        super(ti_template, self).__init__(
            dtype, ndim, element_dim, element_shape, field_dim, needs_grad, boundary
        )

    def __to_taichi_type__(self):
        import taichi as ti
        return ti.types.ndarray(*self.args, **self.kwargs)
