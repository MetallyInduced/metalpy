from metalpy.mexin.utils.type_map import assert_all_types_exists
from metalpy.scab.fixed.fixed import FixedContext


def test_fixed_impls_map():
    assert_all_types_exists(FixedContext._implementations)
