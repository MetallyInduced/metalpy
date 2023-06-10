import re


def replace_batch(string, mapping):
    keys = (re.escape(k) for k in mapping.keys())
    pattern = re.compile('(' + '|'.join(keys) + ')')
    result = pattern.sub(lambda x: mapping[x.group()], string)

    return result
