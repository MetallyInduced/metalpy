from . import Distributable, NotDistributable


def get_patch_policy(patch):
    if isinstance(patch, Distributable):
        return patch
    else:
        return NotDistributable()
