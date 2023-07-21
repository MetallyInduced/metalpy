from __future__ import annotations

import re
import warnings
from typing import Literal, Callable


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
