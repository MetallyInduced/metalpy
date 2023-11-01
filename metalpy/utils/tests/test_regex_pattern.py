import re

import pytest

from metalpy.utils.regex_pattern import expand_regex_pattern
from metalpy.utils.string import format_string_list


def check_expansion(pat, assert_size=None, limit=None):
    try:
        re_pat = re.compile(r'(?!\\)\.').sub(r'\.', pat)
        pattern = re.compile(re_pat)
    except re.error as _:
        with pytest.raises(SyntaxError) as exc_info:
            exc_info.message = f'Inconsistent behavior between `re.compile` and `RegexPattern` for `{pat}`.'
            for _ in expand_regex_pattern(pat, limit=1):
                pass
        return

    if assert_size is not None:
        # 检查展开后的字符串数量是否符合预期
        limit = assert_size + 2
        expansion = expand_regex_pattern(pat, limit=limit)
        expansion = tuple(expansion)

        if len(expansion) >= limit:
            at_least = 'at least '
        else:
            at_least = ''

        # 看起来pytest只会对非简单 assert 输出额外信息
        satisfied = len(expansion) == assert_size
        assert satisfied, (
            f'Unexpected expansion count (got {at_least}{len(expansion)}, expected {assert_size})'
            f' for `{pat}`. Got:'
            f'{format_string_list(expansion, multiline=True, limit=assert_size + 1)}'
        )
    else:
        expansion = expand_regex_pattern(pat, limit=limit)
        if limit is not None:
            # 设定了limit，顺便检查一下limit是否有效
            expansion = tuple(expansion)
            assert len(expansion) <= limit, f'Failed to set `limit` for `{pat}`.'

    for result in expansion:
        assert pattern.fullmatch(result), (
            f'Inconsistent result between `re.compile` and `RegexPattern` for `{pat}`.'
            f' Got `{result}`.'
        )


def test_regex_pattern_range():
    check_expansion(r'\[d-b]', assert_size=1)  # 转义范围符号
    check_expansion(r'[d-b]')  # 错误的范围
    check_expansion(r'[\b-b]')  # 空格到b
    check_expansion(r'[b-\b]')  # 错误的范围
    check_expansion(r'[b-\d]')  # 错误的范围
    check_expansion(r'[\\\-\d-b]')  # 范围内转义
    check_expansion(r'[\db]')  # 转义范围


def test_regex_pattern_repeat():
    check_expansion(r'(a|b)+', limit=10)
    check_expansion(r'(a|b)*', limit=3)
    check_expansion(r'(a|b)?', limit=3)
    check_expansion(r'a{0,3}', assert_size=4)
    check_expansion(r'a{,3}', assert_size=4)
    check_expansion(r'a{3,}', limit=3)


def test_regex_pattern_comprehensive():
    check_expansion('test[1234].(com|cn)/[012]{0,3}', limit=20)

    pat = r'https://webrd0[1-4].is.autonavi.com/appmaptile\?x=\{x}&y=\{y}&z=\{z}'
    check_expansion(pat, assert_size=4)
