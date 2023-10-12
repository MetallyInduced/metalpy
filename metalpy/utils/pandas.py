def drop_by_indices(df, indices, ext=0):
    import pandas as pd

    dropped = df.copy()
    length = max(dropped.index)
    for idx in indices:
        start, end = min(idx), max(idx)
        dropped.drop(index=idx, inplace=True, errors='ignore')
        if ext > 0:
            dropped.drop(index=pd.RangeIndex(max(min(idx) - ext, 0), start), inplace=True, errors='ignore')
            dropped.drop(index=pd.RangeIndex(end, min(max(idx) + ext, length)), inplace=True, errors='ignore')

    return dropped
