import numpy as np
from discretize import TensorMesh, TreeMesh

from metalpy.utils.dhash import dhash, register_dhasher


def dhash_discretize_mesh(mesh: TensorMesh):
    return dhash(mesh)


@register_dhasher(TensorMesh)
def dhash_tensor_mesh(mesh: TensorMesh):
    origin = mesh.origin
    hs = mesh.h

    return dhash(*origin, *hs)


@register_dhasher(TreeMesh)
def dhash_tensor_mesh(mesh: TreeMesh, n_samples=10):
    origin = mesh.origin
    hs = mesh.h
    indices = mesh.cell_state['indexes']
    levels = mesh.cell_state['levels']

    n_tree_cells = len(levels)
    n_samples = min(n_samples, n_tree_cells)
    sample_indices = np.linspace(0, n_tree_cells - 1, n_samples).astype(int)

    return dhash(
        *origin,
        *hs,
        n_tree_cells,
        *[indices[i] for i in sample_indices],
        [levels[i] for i in sample_indices]
    )
