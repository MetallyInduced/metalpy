import enum
from typing import Union


class ArrayType(enum.Enum):
    numpy = 'numpy'
    pandas = 'pandas'

    @staticmethod
    def test(arr):
        module_name = type(arr).__module__
        for atype in ArrayType:
            if module_name.startswith(atype.value):
                return atype


ArrayModuleType = Union[ArrayType, str]


class OperatorDispatcher:
    def __init__(self, op_name: str):
        self.op_name: op_name = op_name
        self.ops: dict[str, callable] = {}

    def register(self, module_name: ArrayModuleType):
        def wrapper(func):
            if isinstance(module_name, ArrayType):
                name = module_name.value
            else:
                name = module_name
            self.ops[name] = func
            return func

        return wrapper

    def alias(self, module_name, alias):
        self.register(module_name)(
            OperatorDispatcher.make_alias_wrapper(module_name, alias)
        )

        return self

    def aliases(self, aliases: dict[ArrayType, str]):
        for module_name, alias in aliases.items():
            self.alias(module_name, alias)

        return self

    register_aliases = aliases

    def dispatch(self, arr_type):
        module_name = arr_type.__module__
        for k, func in self.ops.items():
            if module_name.startswith(k):
                return func

        raise ValueError(f'Unsupported operator {self.op_name} for `{arr_type.__name__}`.')

    @staticmethod
    def make_alias_wrapper(module_name, op_name):
        def wrapper(*args, **kwargs):
            assert isinstance(module_name, ArrayType), \
                'For safety concern, aliases are only allowed in registered packages.'
            func = OperatorDispatcher.import_from_module(module_name.value, op_name)
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def import_from_module(module_name, op_name):
        module = __import__(module_name, fromlist=(op_name,))
        return getattr(module, op_name)
