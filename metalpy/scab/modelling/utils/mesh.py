import warnings

import numpy as np

from metalpy.utils.bounds import Bounds


def is_inside_cuboid(mesh, corner, lengths):
    warnings.warn('`is_inside_cuboid` is deprecated due to performance issue,'
                  ' use `is_inside_bounds` instead.')

    axes_mask = np.zeros(mesh.shape[1], dtype=bool)
    axes_mask[:len(corner)] = ~np.isnan(corner)
    axes_mask[:len(lengths)] &= ~np.isnan(lengths)
    corner_mask = axes_mask[:len(corner)]
    lengths_mask = axes_mask[:len(lengths)]

    if axes_mask.any():
        deltas = mesh[:, axes_mask] - corner[corner_mask]
        return np.all(deltas >= 0, axis=1) & np.all(deltas <= lengths[lengths_mask], axis=1)
    else:
        return np.full(mesh.shape[0], True)


def is_inside_bounds(mesh_points, bounds):
    mesh_points2d = np.atleast_2d(mesh_points)

    bounds = Bounds(bounds)
    mask = np.ones(mesh_points2d.shape[0], dtype=bool)

    for a in range(bounds.n_axes):
        amin, amax = bounds.get(a)
        ax = mesh_points2d[:, a]
        if not np.isnan(amax):
            mask &= ax <= amax
        if not np.isnan(amin):
            mask &= ax >= amin

    if np.ndim(mesh_points) == 1:
        return mask[0]
    else:
        return mask
