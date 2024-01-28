from __future__ import annotations

import collections
import itertools
import warnings
from typing import Collection, Iterable


class RegexPattern:
    """接受正则表达式语法的一个子集（不支持 `.` ， `[^a]` 等任意能匹配字符的语法），
    展开为其所代表的字符串集合

    Examples
    --------
    'aaa|bbb|ccc' 展开为 ['aaa', 'bbb', 'ccc']

    'test[1234].(com|cn)/[012]{0,3}'

    Notes
    -----
    由于不同通配符展开结果会以笛卡尔积形式组合，警惕组合爆炸。

    [...]中的^会自动转义为该字符，不代表取反意义。

    `\\\\`依然为转义字符。

    字符集不包含大写字母表示的补集，例如 `\\S` 等。

    References
    ----------
    从字符串集合生成正则表达式
    http://regex.inginf.units.it/
    """
    EMPTY = ''

    def __init__(self, pattern: str, *, limit: int | bool | None = None, wide_spaces=False):
        self.pattern = pattern
        self.index = 0
        self.n_chars = len(self.pattern)

        if limit is True:
            limit = 100
        self.limit = limit

        self.char_classes = assemble_char_classes(wide_spaces=wide_spaces)

    def __iter__(self):
        if self.limit:
            yield from itertools.islice(self.parse(), self.limit)
        else:
            yield from self.parse()

    def fetch(self, escape=True):
        c = self.pattern[self.index]
        self.index += 1

        if c == '\\' and escape:
            c = self.escape_next()

        return c

    def escape_char_classes(self, c):
        return self.char_classes.get(c, None)

    def escape_next(self):
        c = self.fetch(escape=False)
        ret = self.escape_char_classes(c)
        if ret is None:
            ret = c
        return ret

    def parse(self, level=0):
        options = OrOp()
        components = JoinOp()

        while self.index < self.n_chars:
            c = self.fetch(escape=False)

            if c == '\\':
                components.append_char_or_component(self.escape_next())
            elif c == '(':
                components.seal_chars()
                components.append(self.parse(level=level + 1))
            elif c == '[':
                components.seal_chars()
                components.append(self.parse_charset())
            elif c in ('{', '?', '+', '*'):
                repeated = components.pop_last_char_and_seal_rest()
                if repeated is None:
                    if len(components) == 0:
                        raise SyntaxError('Repeat operator on empty operand.')
                    repeated = components.pop()
                components.append(self.parse_repetition(component=repeated, op=c))
            elif c == '|':
                components.seal_chars()
                options.append(components.unwrap())
                components = JoinOp()
            elif c == ')':
                if level <= 0:
                    raise SyntaxError('Brackets not enclosed.')
                break
            else:
                components.append_char(c)

        components.seal_chars()
        options.append(components.unwrap())
        return options.unwrap()

    def parse_charset(self):
        ret = CharsetOp()
        last = None

        while self.index < self.n_chars:
            c = self.fetch()

            if c == '-':
                c = self.fetch()

                if not isinstance(last, str) or not isinstance(c, str):
                    # 如果不是字符，那就是 `[` 或 `CharsetOp` ，是字符类转义
                    # 不应被 `-` 视为范围起点
                    start = last
                    if isinstance(start, CharsetOp):
                        start = start.clazz
                    elif start is None:
                        start = '['

                    end = c
                    if isinstance(end, CharsetOp):
                        end = end.clazz

                    raise SyntaxError(f'Unexpected `-` encountered between `{start}` and `{end}`.')

                if ord(last) > ord(c):
                    raise SyntaxError(f'Bad character range `{last}` and `{c}`.')
                ret.extend(chr(i) for i in range(ord(last), ord(c) + 1))
            elif c != ']':
                if c == char_class_empty:
                    # TODO: 明确这个搭配的语义
                    # \b 出现在范围语法中，似乎是视为空格
                    c = ' '
                ret.append_char_or_component(c)
            else:
                break

            last = c

        return ret

    def parse_repetition(self, component, op):
        if op == '{':
            values = []
            value = None
            while self.index < self.n_chars:
                c = self.fetch()

                if c == ',':
                    values.append(value)
                    value = None
                elif c == ' ':
                    continue
                elif c == '}':
                    values.append(value)
                    break
                else:
                    try:
                        v = int(c)

                        if value is None:
                            value = v
                        else:
                            value = value * 10 + v
                    except ValueError:
                        raise SyntaxError(f'Unexpected char `{c}` encountered.')

            if len(values) == 1:
                mmin, mmax = values[0], values[0]
            elif len(values) == 2:
                mmin, mmax = values
            else:
                raise SyntaxError(f'Invalid use of repeat.')

            if mmin is None:
                mmin = 0
        elif op == '?':
            mmin, mmax = 0, 1
        elif op == '+':
            mmin, mmax = 1, None
        elif op == '*':
            mmin, mmax = 0, None
        else:
            raise SyntaxError(f'Unknown repeat operator `{op}`.')

        if mmax is None and self.limit is None:
            warnings.warn(f'Unlimited quantifier `{op}` detected at {self.index},'
                          f' consider specifying `limit=100` to limit the output'
                          f' or `limit=False` to explicitly mute the warning message.')

        return RepeatOp(component, mintimes=mmin, maxtimes=mmax)


def expand_regex_pattern(pattern: str, limit: int | bool | None = None):
    pattern = RegexPattern(pattern, limit=limit)
    return pattern


class LiteralOp:
    def __init__(self, literal):
        self.literal = literal

    def __iter__(self):
        yield self.literal


class JoinOp:
    def __init__(self):
        self.components = []
        self.chars = ''

    def append(self, component):
        self.components.append(component)

    def pop(self):
        return self.components.pop()

    def append_char(self, c):
        self.chars += c

    def append_char_or_component(self, c):
        if isinstance(c, str):
            self.append_char(c)
        else:
            self.append(c)

    def seal_chars(self):
        if self.chars != '':
            self.append(LiteralOp(self.chars))
            self.chars = ''

    def pop_last_char_and_seal_rest(self):
        n_chars = len(self.chars)
        if n_chars > 1:
            self.append(LiteralOp(self.chars[:-1]))
            ret = self.chars[-1]
            self.chars = ''
        elif n_chars == 1:
            ret = self.chars
            self.chars = ''
        else:
            ret = None

        return ret

    def unwrap(self):
        if len(self.components) == 1:
            return self.components[0]
        else:
            return self

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        for parts in lazy_product(*self.components):
            yield ''.join(parts)


class OrOp:
    def __init__(self):
        self.components = []

    def append(self, composition):
        self.components.append(composition)

    def unwrap(self):
        if len(self.components) == 1:
            return self.components[0]
        else:
            return self

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        for c in self.components:
            yield from c


class CharsetOp:
    def __init__(self, init=None, clazz: str | None = None):
        if clazz is not None:
            if not clazz.startswith('\\'):
                clazz = '\\' + clazz

            if len(clazz) > 2:
                warnings.warn(f'`{CharsetOp.__name__}` got `{clazz}` as class identifier,'
                              f' taking it as {clazz[:2]}.')
                clazz = clazz[:2]
        self.clazz = clazz
        self.chars = collections.OrderedDict()
        if init is not None:
            self.extend(init)

    @property
    def class_identifier(self):
        return self.clazz[1:]

    def append(self, char):
        self.chars[char] = None

    def extend(self, string):
        for char in string:
            self.append(char)

    def append_char_or_component(self, c):
        if isinstance(c, str):
            self.extend(c)
        elif isinstance(c, CharsetOp):
            self.extend(c.chars)

    def __len__(self):
        return len(self.chars)

    def __iter__(self):
        yield from self.chars

    def __eq__(self, other):
        if not isinstance(other, CharsetOp):
            return False
        return self.clazz == other.clazz and self.chars == other.chars


class RepeatOp:
    def __init__(self, component, maxtimes: int | None, mintimes: int = 0):
        self.component = component
        self.mintimes = mintimes
        self.maxtimes = maxtimes

    def __iter__(self):
        base = [self.component]
        times = self.mintimes

        while self.maxtimes is None or times <= self.maxtimes:
            if times == 0:
                yield ''
            else:
                for parts in lazy_product(*(base * times)):
                    yield ''.join(parts)

            times += 1


def lazy_product(*iterables, all_collections=False):
    """真正的lazy版product

    `itertools.product` 要求在迭代前就将所有迭代器遍历一遍得到 `tuple` 序列，
    这样会导致无穷序列无法适用于此操作。

    而 `lazy_product` 则通过单独实现一个缓存，来实现扫描的同时创建序列。

    Parameters
    ----------
    iterables
        用于求解笛卡尔积的迭代器
    all_collections
        指示是否所有的序列都是有穷序列

    Returns
    -------
    cartesian_product
        多个迭代器的笛卡尔积结果
    """
    n_iterables = len(iterables)
    if n_iterables < 1:
        return

    iterable = iterables[0]
    if n_iterables == 1:
        yield from iterable
    elif n_iterables == 2 and isinstance(iterable, Collection):
        for elem0 in iterable:
            for elem1 in iterable:
                yield elem0, elem1
    else:
        others = iterables[1:]
        if all_collections or check_if_all_collections(others):
            for elem0 in iterable:
                for elems in lazy_product(*iterables[1:], all_collections=True):
                    yield elem0, *elems
        else:
            elements = []
            for elem0 in iterable:
                if len(elements) == 0:
                    for elems in lazy_product(*iterables[1:]):
                        elements.append(elems)
                        yield elem0, *elems
                else:
                    for elems in elements:
                        yield elem0, *elems


def check_if_all_collections(iterables: Iterable[Iterable]):
    return all(isinstance(iterable, Collection) for iterable in iterables)


spaces7 = ' \r\n\t\v\f'
spaces16 = ''.join([chr(c) for c in [
    160, 8232, 8233, 5760, 6158, 8192, 8193, 8194, 8195, 8196,
    8197, 8198, 8199, 8200, 8201, 8202, 8239, 8287, 12288, 65279
]])
numbers = ''.join([chr(i + ord('0')) for i in range(10)])
letters = ''.join([chr(i + ord('a')) for i in range(26)])

char_class_numbers = CharsetOp(init='0123456789', clazz=r'd')
char_class_spaces7 = CharsetOp(init=spaces7, clazz=r's')
char_class_spaces16 = CharsetOp(init=spaces7 + spaces16, clazz=r's')
char_class_identifiers = CharsetOp(init='_' + letters + letters.upper() + numbers, clazz=r'w')
char_class_empty = CharsetOp(init='', clazz=r'b')


def assemble_char_classes(wide_spaces=False):
    classes = [
        char_class_numbers,
        char_class_identifiers,
        char_class_empty
    ]
    if wide_spaces:
        classes.append(char_class_spaces16)
    else:
        classes.append(char_class_spaces7)

    return {
        cls.class_identifier: cls for cls in classes
    }
