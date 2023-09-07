from __future__ import annotations

import re
import warnings
from typing import Literal, Callable, Iterable, TypeVar, Sequence

_T = TypeVar('_T')


def replace_batch(string, mapping):
    keys = (re.escape(k) for k in mapping.keys())
    pattern = re.compile('(' + '|'.join(keys) + ')')
    result = pattern.sub(lambda x: mapping[x.group()], string)

    return result


def format_string_list(strs,
                       multiline: int | bool = 15,
                       limit: int | Literal[False] = 5,
                       line_indent=4,
                       quote: str | Callable | None = None):
    """将给定字符串数组格式化为列表形式

    Parameters
    ----------
    strs
        字符串列表
    multiline
        是否允许在字符串元素较长时格式化为多行列表，也可通过指定一个数值来限制换行的大小
    limit
        最大将输出的元素个数，False代表不限制输出元素个数
    line_indent
        触发格式化为多行列表时，行的缩进字符数
    quote
        对每个字符串两端添加字符，None代表不作处理。
        若为字符串，长度为1时两端添加相同字符，长度为2时分别加在两端。

    Returns
    -------
    formatted_str
        字符串数组格式化后的结果

    Notes
    -----
    multiline若为True，则采用默认长度限制15；
    若为False，则不会格式化为多行列表；
    若为int，则在元素长度超过该值时格式化为多行列表。

    转化为多行列表时会自动在字符串前添加换行符。

    Examples
    --------
    >>> print(format_string_list(['`Grav.`', '`Mag.`', '`EM`'], multiline=False, limit=1))
    <<< `Grav.` and ... (2 more)

    >>> print(format_string_list(['`Grav.`', '`Mag.`', '`EM`'], multiline=True, limit=1))
    <<<
    <<<     - `Grav.`
    <<<     - ... (2 more)

    >>> print(format_string_list(['`Grav.`', '`Mag.`', '`EM`'], multiline=False, limit=False))
    <<< `Grav.`, `Mag.` and `EM`

    >>> print(format_string_list(['`Grav.`', '`Mag.`', '`EM`'], multiline=True, limit=False))
    <<<
    <<<     - `Grav.`
    <<<     - `Mag.`
    <<<     - `EM`
    """
    if quote is not None:
        if isinstance(quote, str):
            if len(quote) == 1:
                lq, rq = quote, quote
            elif len(quote) == 2:
                lq, rq = quote
            else:
                warnings.warn(f'`format_string_list` with more than two `quote`s: {quote}')
                lq, rq = quote[:2]
            strs = [lq + s + rq for s in strs]
        elif callable(quote):
            strs = [quote(s) for s in strs]
        else:
            warnings.warn(f'`format_string_list` got unknown `quote`: {quote}.'
                          f' Expect `str` or `Callable`.'
                          f' Ignoring it.')
    else:
        if not isinstance(strs, list):
            strs = list(strs)

    if not isinstance(multiline, bool):
        line_limit = multiline
        if line_limit >= 0:
            max_length = max([len(s) for s in strs])
            multiline = max_length > line_limit
        else:
            multiline = True

    formatted = strs
    if limit:
        n_omitted = len(strs) - limit
        if n_omitted > 0:
            formatted = [*strs[:limit], f'... ({n_omitted} more)']

    if multiline:
        prefix = ' ' * line_indent + '- '
        return '\n' + '\n'.join([prefix + s for s in formatted])
    else:
        if len(formatted) > 1:
            return ', '.join(formatted[:-1]) + ' and ' + formatted[-1]
        else:
            return formatted[0]


def parse_axes_labels(
        labels: Iterable[str | int] | int,
        max_length: int | None = None,
        length: int | None = None
):
    return parse_labels(labels=labels, accepts='xyz', max_length=max_length, length=length)


def parse_labels(
        labels: Iterable[_T | int] | _T | int,
        accepts: Sequence[_T],
        max_length: int | None = None,
        length: int | None = None
):
    """解析一组字符串或数组为下标数组，例如将 'xzy' 解析为 [0, 2, 1]。

    Parameters
    ----------
    labels
        待解析的labels
    accepts
        指定接受的字符列表
    max_length
        指定接受的最大标签数
    length
        指定接受的标签数，多余或少于均拒绝

    Returns
    -------
    parsed_indices
        根绝给定的字符或整数列表解析的下标列表

    Examples
    --------
    >>> print(parse_axes_labels('xzy'))
    <<< [0, 2, 1]
    """
    max_idx = len(accepts)
    error_msg = f'`parse_axes_labels` accepts only' \
                f' {format_string_list(accepts, multiline=False)}' \
                f' or integer number < {max_idx}. Got `{{got}}`.'

    if length is not None:
        max_length = length

    if not isinstance(labels, Iterable):
        labels = (labels,)

    ret = []

    for c in labels:
        if isinstance(c, int):
            idx = c
        else:
            try:
                idx = accepts.index(c)
            except ValueError:
                idx = max_idx  # 找不到喵
        assert idx < max_idx, error_msg.format(got=c)
        ret.append(idx)

        if max_length is not None:
            assert len(ret) <= max_length, 'Too many labels.'

    if length is not None:
        assert len(ret) == length, f'Exactly {length} label(s) are required.'

    return ret
