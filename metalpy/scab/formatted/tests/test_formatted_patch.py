from metalpy.mexin.utils.type_map import assert_all_types_exists
from metalpy.scab.formatted.formatted import FormattedContext


def test_formatted_impls_map():
    assert_all_types_exists(FormattedContext._implementations)
