from metalpy.mexin.utils.type_map import assert_all_types_exists
from metalpy.scab.progressed.progressed import Progress


def test_progressed_impls_map():
    assert_all_types_exists(Progress._implementations)
