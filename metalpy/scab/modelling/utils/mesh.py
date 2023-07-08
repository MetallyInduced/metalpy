import numpy as np


def is_inside_cuboid(mesh, corner, lengths):
    axes_mask = np.full(mesh.shape[1], False)
    axes_mask[:len(corner)] = ~np.isnan(corner)
    axes_mask[:len(lengths)] &= ~np.isnan(lengths)

    if axes_mask.any():
        deltas = mesh[:, axes_mask] - corner[axes_mask]
        return np.all(deltas >= 0, axis=1) & np.all(deltas <= lengths[axes_mask], axis=1)
    else:
        return np.full(mesh.shape[0], True)
