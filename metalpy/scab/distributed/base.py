from __future__ import annotations

import functools

import blosc2
from SimPEG.simulation import BaseSimulation
from discretize import TensorMesh

from metalpy.mepa.utils import structured_traverse
from metalpy.mexin import Mixin
from .policies import get_patch_policy
from . import Distributed


class BaseDistributedSimulationMixin(Mixin):
    def __init__(self, this, patch: Distributed):
        super().__init__(this)
        self.patch = patch

    @property
    def executor(self):
        return self.patch.executor

    @property
    def persisted_context(self):
        return self.patch.persisted_context

    @property
    def patches(self):
        return self.persisted_context.get_patches()

    def compress_mesh(self, this: BaseSimulation | None = None):
        from pympler.asizeof import asizeof

        mesh = this.mesh
        h = list(mesh.h)

        if not self.executor.needs_serialization() or asizeof(h) < 3000:
            return functools.partial(_uncompress_direct, mesh)
        else:
            if isinstance(mesh, TensorMesh):
                return functools.partial(_uncompress_mesh, compress_array(list(h)), mesh.origin)
            else:
                raise NotImplementedError(f'Only `TensorMesh` is supported, got `{type(mesh)}`.')

    def distribute_patches(self, _=None):
        patches_list = []
        for w in self.executor.get_workers():
            patches = [
                get_patch_policy(patch).distribute_to(self.executor, w)
                for patch in self.patches
            ]
            patches = [patch for patch in patches if patch is not None]
            patches_list.append(patches)

        return patches_list


def compress_array(arr, direct=False):
    if direct:
        return functools.partial(_uncompress_direct, arr)
    else:
        compressed = structured_traverse(arr, lambda x: blosc2.pack_array2(x))
        return functools.partial(_uncompress, compressed)


def _uncompress_direct(c):
    return c


def _uncompress(c):
    return structured_traverse(c, lambda x: blosc2.unpack_array2(x))


def _uncompress_mesh(h, origin):
    return TensorMesh(
        h=h(),
        origin=origin
    )
