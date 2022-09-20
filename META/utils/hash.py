

# return the hash of the given string in string format
def hash_str(*objs):
    flattened = []
    for obj in objs:
        if isinstance(obj, list):
            flattened.extend(obj)
        else:
            flattened.append(obj)

    flattened = tuple(flattened)
    char_table = '0123456789abcdef'
    tablesize = len(char_table)
    hash_value = hash(flattened)

    hash_str = ''

    while hash_value != 0 and len(hash_str) < 6:
        c = hash_value % tablesize
        hash_str = char_table[c] + hash_str
        hash_value = hash_value // tablesize

    return hash_str
