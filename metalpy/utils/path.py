from typing import Iterable

from metalpy.utils.string import replace_batch

windows_forbidden_chars = r'\/:*?"<>|'


def to_url_escape_sequence(c):
    if c == '%':
        return '%%'
    else:
        return '%{:02X}'.format(ord(c))


def get_char_escape_mapping(chars: Iterable[str], inverse=False):
    if not inverse:
        return {c: to_url_escape_sequence(c) for c in set(chars) | set('%')}
    else:
        return {to_url_escape_sequence(c): c for c in set(chars) | set('%')}


def get_char_escape_trans(chars: Iterable[str], inverse=False):
    return str.maketrans(get_char_escape_mapping(chars, inverse=inverse))


def pathencode(path, forbidden=windows_forbidden_chars):
    path = replace_batch(path, get_char_escape_mapping(forbidden))
    return path


def pathdecode(path, forbidden=windows_forbidden_chars):
    path = replace_batch(path, get_char_escape_mapping(forbidden, inverse=True))
    return path
