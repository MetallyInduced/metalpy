"""
Notes
-----
    为了防止多余的import占用运行时间，因此保持只在必要时import相关类型，
    但是缺陷是每次添加新的Builder时，需要进行如下三步操作：

    - `TYPE_CHECKING`中引入类型
    - `SimulationBuilder`中定义`of`的重载
    - 使用__register_builder注册实际类型

    其中前两步用于明确`SimulationBuilder.of`的返回值，第三步为实际注册对应Simulation的builder类。

    `supplier`为builder系统中用于指定参数值的方法，
    实现方式为使用`_supplies`注解，自动将返回值赋给注解中注册的参数。

    `assembler`为builder系统中用于从给定参数构造为给定参数的默认值的方法，
    会在build时对所有未给定的必选参数调用对应的assembler。
    实现方式为使用`_assembles`注解，将函数名注册到基类的_all_assemblers字典中，
    __new__时根据注册的类型名和函数名，获取实例的成员函数构建实例的_assemblers字典。
"""
import functools
import inspect
import warnings
from functools import lru_cache
from typing import overload, TYPE_CHECKING, cast

from metalpy.scab import simpeg_patched

from metalpy.mexin import Patch

from metalpy.mexin.utils import TypeMap
from metalpy.mexin.utils.misc import reget_object
from metalpy.utils.arg_specs import ArgSpecs
from metalpy.utils.object_path import ObjectPath, get_full_qualified_path
from metalpy.utils.type import undefined
from .utils import MissingArgsException, MissingArgException

if TYPE_CHECKING:
    # 用于类型检查时前向声明，运行时`TYPE_CHECKING` == False，import代码不会执行
    from SimPEG.potential_fields import magnetics
    from .potential_fields import magnetics as _magnetics


class SimulationBuilder:
    _registry = TypeMap(allow_match_parent=True)
    _all_assemblers = {}
    _all_suppliers = {}

    def __init__(self, sim_cls):
        """用于构造正演仿真

        Parameters
        ----------
        sim_cls
            正演仿真计算的类
        """
        self.sim_cls = sim_cls
        self.args = ArgSpecs.from_class_mro(self.sim_cls)
        self._assemblers = {}
        self._patches: list[Patch] = []

    def patched(self, *patches: Patch):
        self._patches.extend(patches)

    def build(self):
        with simpeg_patched(*self._patches):
            for name, assembler in self.get_assemblers():
                key = self.args.find_arg_key(name)
                if key is None:
                    continue

                try:
                    self.args.get_bound_arg(key)
                except KeyError:
                    self.args.bind_arg(key, assembler())

            try:
                ret = self.args.call(reget_object(self.sim_cls))
                return ret
            except ValueError as e:
                ex = MissingArgsException()
                for arg in e.args[1] + e.args[2]:
                    ex.append(self.report_missing(arg.name))
                raise ex

    def report_missing(self, name, methods=None):
        return MissingArgException(name, methods if methods is not None else self.get_suppliers(name))

    def __new__(cls, sim_cls: type):
        builder_cls = cast(type[SimulationBuilder], SimulationBuilder._registry.get(sim_cls))
        if builder_cls is None:
            raise NotImplementedError(f'SimulationBuilder does not support {sim_cls.__name__} for now.')
        ret = super(SimulationBuilder, cls).__new__(builder_cls)
        return ret

    @staticmethod
    @overload
    def of(sim_cls: 'type[magnetics.Simulation3DIntegral]') -> '_magnetics.Simulation3DIntegralBuilder':
        ...

    @staticmethod
    def of(sim_cls):
        return SimulationBuilder(sim_cls)

    @staticmethod
    def _implies(*keysets):
        keysets = tuple((keyset,) if isinstance(keyset, str) else keyset for keyset in keysets)

        def decorator(func):
            cls_name = str(ObjectPath.of(func).parent)
            suppliers = SimulationBuilder._all_suppliers.setdefault(cls_name, {})

            name = getattr(func, '__name__', None)
            if name is None:
                warnings.warn(f'Unnamed function will not be notified if missing: {ObjectPath.of(func)}.')
            else:
                for keyset in keysets:
                    for key in keyset:
                        suppliers.setdefault(key, []).append(func.__name__)

            return func

        return decorator

    @staticmethod
    def _supplies(*keysets, allow_vars=False):
        """指示该函数为对应参数的提供器。用户调用该函数时会为对应形参提供实参。
        所修饰的函数会通过返回值为给定参数名提供参数，并会被包装为返回self的链式调用

        Parameters
        ----------
        keysets
            给定关键则，长度与返回值数匹配，每个元素可以是单个或者若干个参数名。
            指示将对应位置的返回值绑定到对应的所有参数名下。
        allow_vars
            指示是否允许在目标函数固定参数中找不到对应参数时，将其绑定到目标的varkw上，默认为False

        Returns
        -------
        wrapper
            返回的组装器函数包装器，负责将修饰的函数注册为对应参数的提供者

        Notes
        -----
        如果找不到给定的参数名，且allow_vars为False或目标不接受varkw，则抛出错误

        参数中已设置的实参会以相同名字的kwarg传递给绑定的函数，
        例如设置一个函数为名为`model_type`的形参提供实参，
        则如果之前该实参有绑定过值，_supplies的wrapper会将该值连同用户输入一并传递给绑定的函数。

        >>>@SimulationBuilder._supplies('model_type')
        >>>def model_type(self, type=None, *, model_type=None):
        >>>    pass

        >>>builder.model_type('scalar')  # model_type的输入: type='scalar'
        >>>builder.model_type('vector')  # model_type的输入: type='vector', model_type='scalar'
        """
        keysets = tuple((keyset,) if isinstance(keyset, str) else keyset for keyset in keysets)

        def decorator(func):
            SimulationBuilder._implies(*keysets)(func)

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                for keyset in keysets:
                    for key in keyset:
                        try:
                            val = self.args.get_bound_arg(key)
                            kwargs[key] = val
                        except KeyError:
                            pass

                ret = func(self, *args, **kwargs)
                if len(keysets) == 1:
                    # 如果指定绑定的参数长度为1，则返回值不要求为形似 `return x,` 的元组形式
                    ret = (ret,)

                assert len(ret) == len(keysets), \
                    "Arg suppliers' return values must have same length as `_supplies` signature."

                for keyset, val in zip(keysets, ret):
                    try:
                        if undefined == val:
                            continue
                        for key in keyset:
                            key = self.args.find_arg_key(key, allow_vars=allow_vars)
                            self.args.bind_arg(key, val)
                    except KeyError:
                        pass

                return self

            return wrapper

        return decorator

    @staticmethod
    def _assembles(*keysets):
        """指示该函数为对应参数的组装器，在指定形参缺失时由Builder系统调用该方法构造实参。

        Parameters
        ----------
        keysets
            给定关键则，长度与返回值数匹配，每个元素可以是单个或者若干个参数名。
            指示将对应位置的返回值绑定到对应的所有参数名下。

        Returns
        -------
        wrapper
            返回的组装器函数包装器，负责将修饰的函数注册为对应参数的组装器
        """
        keysets = tuple((keyset,) if isinstance(keyset, str) else keyset for keyset in keysets)

        def decorator(func):
            cls_name = str(ObjectPath.of(func).parent)
            assemblers = SimulationBuilder._all_assemblers.setdefault(cls_name, {})

            for keyset in keysets:
                for key in keyset:
                    assemblers[key] = func.__name__

            return func

        return decorator

    def get_assemblers(self):
        for cls in reversed(inspect.getmro(type(self))):
            cls_name = get_full_qualified_path(cls)
            for key, assembler_name in SimulationBuilder._all_assemblers.get(cls_name, {}).items():
                yield key, getattr(self, assembler_name)

    @classmethod
    @lru_cache
    def get_suppliers(cls, name):
        suppliers = []
        for sub_cls in reversed(inspect.getmro(cls)):
            cls_name = get_full_qualified_path(sub_cls)
            suppliers.extend(SimulationBuilder._all_suppliers.get(cls_name, {}).get(name, tuple()))

        return suppliers


def __register_builder(key_cls_path):
    def decorator(func):
        SimulationBuilder._registry.map(key_cls_path, func)
        return func
    return decorator


@__register_builder('SimPEG.potential_fields.magnetics.simulation.Simulation3DIntegral')
def _():
    from .potential_fields.magnetics.simulation import Simulation3DIntegralBuilder
    return Simulation3DIntegralBuilder
