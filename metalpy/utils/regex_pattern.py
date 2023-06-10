import collections
import itertools


class RegexPattern:
    """接受正则表达式语法的一个子集（只支持[xyz]、(a|b)、|、?、{m, n}、\\\\d等有限状态），
    展开为其所代表的字符串集合

    Examples
    --------
    'aaa|bbb|ccc' 展开为 ['aaa', 'bbb', 'ccc']

    'test[1234].(com|cn)/[012]{0, 3}'

    Notes
    -----
    由于不同通配符展开结果会以笛卡尔积形式组合，警惕组合爆炸。

    [...]中的^会自动转义为该字符，不代表取反意义。

    `\\\\`依然为转义字符。

    '.'，'*'，'+'等元字符不再作为特殊字符存在，但仍可通过'\\\\'转义，'?'含义不变。

    特殊转义字符只包含\\\\d以及其它标准转义符*，但不会包含\\\\s。

    *其它标准转义符暂未实现

    References
    ----------
    从字符串集合生成正则表达式
    http://regex.inginf.units.it/
    """

    EMPTY = ''

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.index = 0
        self.n_chars = len(self.pattern)

    def __iter__(self):
        yield from self.parse()

    def fetch(self):
        c = self.pattern[self.index]
        self.index += 1
        return c

    def escape(self, c):
        # TODO：实现其它标准转义符\
        if c == 'd':
            return CharsetOp(init='0123456789')
        else:
            return c

    def parse(self, level=0):
        options = OrOp()
        components = JoinOp()
        escape = False

        while self.index < self.n_chars:
            c = self.fetch()

            if c == '\\':
                escape = True
            elif escape:
                escape = False
                components.append_char_or_component(self.escape(c))
            elif c == '(':
                components.seal_chars()
                components.append(self.parse(level=level + 1))
            elif c == '[':
                components.seal_chars()
                components.append(self.parse_charset())
            elif c == '{' or c == '?':
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
        escape = False

        while self.index < self.n_chars:
            c = self.fetch()

            if c == '\\':
                escape = True
            elif escape:
                escape = False
                ret.append_char_or_component(self.escape(c))
            elif c != ']':
                ret.append(c)
            else:
                break

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
                    except ValueError:
                        raise SyntaxError(f'Unexpected char `{c}` encountered.')

                    if value is None:
                        value = v
                    else:
                        value = value * 10 + v

            if len(values) == 1:
                mmin, mmax = values[0], values[0]
            elif len(values) == 2:
                mmin, mmax = values
            else:
                raise SyntaxError(f'Invalid use of repeat.')
        elif op == '?':
            mmin, mmax = 0, 1
        else:
            raise SyntaxError(f'Unknown repeat operator `{op}`.')

        return RepeatOp(component, mintimes=mmin, maxtimes=mmax)


def expand_regex_pattern(pattern: str):
    pattern = RegexPattern(pattern)
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
        for parts in itertools.product(*self.components):
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
    def __init__(self, init=None):
        self.chars = collections.OrderedDict()
        if init is not None:
            self.extend(init)

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


class RepeatOp:
    def __init__(self, component, maxtimes, mintimes=0):
        self.component = component
        self.mintimes = mintimes
        self.maxtimes = maxtimes

    def __iter__(self):
        base = [self.component]
        for times in range(self.mintimes, self.maxtimes + 1):
            for parts in itertools.product(*(base * times)):
                yield ''.join(parts)
