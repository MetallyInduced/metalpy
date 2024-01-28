import abc
import inspect


class ABCDuckingMeta(abc.ABCMeta):
    def __instancecheck__(self, instance):
        if super().__instancecheck__(instance):
            # 检查是否为直接或间接子类的实例
            return True
        else:
            # 检查是否为虚拟子类
            # 检索所有虚函数，如果某个实例可以访问所有虚函数，则可以视为是虚子类
            # 包括成员函数、静态函数和类函数，无论以什么形式提供
            for method in self.__all_user_members__:
                if not hasattr(instance, method):
                    return False
            else:
                return True

    @property
    def __all_user_members__(cls):
        """获取类型中用户定义的成员（包括虚函数和非虚成员）
        """
        func_names = set()

        for clz in inspect.getmro(cls):
            if isinstance(clz, ABCDuckingMeta):
                # 只检查ABCDuckingMeta影响下的基类成员
                private_prefix = f'_{clz.__name__}__'
                members = [k for k in clz.__dict__.keys() if not k.startswith(private_prefix)]
                func_names.update(members)

        # 移除因ABCDucking本身而引入的额外成员
        func_names.difference_update(ABCDucking.__dict__.keys())

        return func_names


class ABCDucking(metaclass=ABCDuckingMeta):
    """实现基于虚基类的通用虚子类检查机制 （鸭子类型）

    子类既可以继承虚基类，也可以直接实现虚函数，
    使用者均可以通过 isinstance() 或 issubclass() 检查类型或实例是否为虚子类
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        if cls in inspect.getmro(subclass):
            # 检查是否为直接或间接子类
            return True
        else:
            # 检查是否为虚拟子类
            # 检索所有虚函数，如果某个子类实现了所有虚函数，则可以视为是虚子类
            # 包括成员函数、静态函数和类函数，但必须以相同的形式提供
            # 例如虚静态函数必须以静态函数的形式提供，虚成员函数必须以成员函数的形式提供
            for method in cls.__all_user_members__:
                meth = inspect.getattr_static(cls, method, None)
                if meth is None:
                    continue

                subclass_meth = inspect.getattr_static(subclass, method, None)
                if subclass_meth is None:
                    return False

                if type(meth) is not type(subclass_meth):
                    return False
            else:
                return True


ABCD = ABCDucking
