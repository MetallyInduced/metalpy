import pytest

from metalpy.mexin.mixins import DispatcherMixin
from metalpy.mexin.utils.type_map import assert_all_types_exists
from metalpy.scab import *  # 载入所有Patch，同时也载入了scab模块内所有DispatcherMixin的子类
from metalpy.utils.type import get_all_subclasses

mixins = get_all_subclasses(DispatcherMixin)


def _():
    simpeg_patched()  # 使用一下 `metalpy.scab` 内的东西，防止警告


@pytest.mark.parametrize('mixin_type', mixins, indirect=False)
def test_dispatcher(mixin_type: DispatcherMixin):
    assert_all_types_exists(mixin_type._impls)
