from typing import TypeVar, Union

from metalpy.utils.type import get_all_subclasses

T = TypeVar('T', covariant=True)
T2 = TypeVar('T2', covariant=True)


class OrderedTypesetMeta(type):
    """
    使用 `|` 来创建联合类型，生成两个操作数类型的公共子类

    使用 `==` 或 `is` 来比较联合类型

    用户自行定义的子类如果符合以下条件，则最后定义的子类会作为或运算的结果子类：

    * 子类的所有基类与所有参与或运算的基类类型相同

    * 子类的基类顺序与基类或运算的顺序相同
    """
    prefix = '+Subclass'
    empty_union_name = f'{prefix}[]'

    def __or__(cls: T, other: T2) -> Union[T, T2]:
        bases = tuple()

        if not OrderedTypesetMeta.is_empty(cls):
            if OrderedTypesetMeta.is_union(cls):
                bases += cls.__bases__  # 已经是联合类型，提取基类
            else:
                bases += (cls,)

        if not OrderedTypesetMeta.is_empty(other):
            if OrderedTypesetMeta.is_union(other):
                bases += other.__bases__  # 已经是联合类型，提取基类
            else:
                bases += (other,)

        union_type = OrderedTypesetMeta.find(bases)  # 查找是否存在符合条件的子类
        if union_type is None:
            base_names = [base.__name__ for base in bases]
            qualname = f'{OrderedTypesetMeta.prefix}[{", ".join(base_names)}]'
            union_type = OrderedTypesetMeta(
                qualname,
                bases,
                {'__qualname__': qualname, '__module__': OrderedTypesetMeta.__module__}
            )

        return union_type

    def __hash__(cls):
        return hash(cls.__bases__)

    def __eq__(cls, other):
        if OrderedTypesetMeta.is_union(cls) and OrderedTypesetMeta.is_union(other):
            return cls.__bases__ == other.__bases__
        else:
            return super().__eq__(other)

    @staticmethod
    def find(bases):
        """查找基于给定若干基类派生得到的联合子类，包括用户手动定义的子类

        最后定义符合条件的子类的最优先采用，因此可以支持用户自定义联合子类行为

        Parameters
        ----------
        bases
            指定基类集合

        Returns
        -------
        subclass
            以 `bases` 作为基类的最晚定义的子类，若未找到，则返回None
        """
        for subclass in reversed(get_all_subclasses(bases[0])):
            if subclass.__bases__ == bases:
                return subclass
        else:
            return None

    @staticmethod
    def is_empty(cls):
        return (
                cls.__qualname__ == OrderedTypesetMeta.empty_union_name
                and len(cls.__bases__) == 1
                and cls.__bases__[0] is object
        )

    @classmethod
    def is_union(cls, typ):
        return cls.prefix in typ.__qualname__


union = OrderedTypesetMeta(
    OrderedTypesetMeta.empty_union_name,
    tuple(),
    {'__qualname__': OrderedTypesetMeta.empty_union_name, '__module__': OrderedTypesetMeta.__module__}
)


class OrderedTypeset(metaclass=OrderedTypesetMeta):
    pass
