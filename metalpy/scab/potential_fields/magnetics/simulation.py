from typing import Literal

import numpy as np
from SimPEG import props, maps
from SimPEG.potential_fields import magnetics
from SimPEG.potential_fields.base import BasePFSimulation
from scipy.constants import mu_0, pi


class Simulation3DDipoles(magnetics.Simulation3DIntegral):

    moment, momentMap, momentDeriv = props.Invertible("Magnetic Moment (A m m)")  # 磁矩

    def __init__(
        self,
        source_locations,
        momentMap=None,
        is_amplitude_data=False,
        store_sensitivities: Literal['ram', 'disk', 'forward_only'] = 'ram',
        sensitivity_dtype=np.float32,
        n_processes=1,
        **kwargs
    ):
        self.store_sensitivities = store_sensitivities
        self.sensitivity_dtype = sensitivity_dtype

        super(BasePFSimulation, self).__init__(None, **kwargs)

        self.solver = None
        self.n_processes = n_processes

        self.solver = None

        self._source_locations = np.atleast_2d(source_locations)
        self.nC = self._source_locations.shape[0]

        if momentMap is None:
            momentMap = maps.IdentityMap(nP=self._source_locations.shape[0] * 3)

        self.momentMap = momentMap
        self._G = None
        self._M = None
        self._gtg_diagonal = None
        self.is_amplitude_data = is_amplitude_data
        self.model_type = 'vector'

    def evaluate_integral(self, receiver_location, components):
        """求解磁偶极子的磁感应强度
        """
        eps = 1e-5

        receiver_location = np.atleast_2d(receiver_location)

        n_obs = receiver_location.shape[0]
        n_dipoles = self._source_locations.shape[0]

        # 1, nC, 3, nObs, 1, 3
        dr = receiver_location[:, np.newaxis, :] - self._source_locations[np.newaxis, :, :]
        dr_norm = np.linalg.norm(dr, axis=2)
        dr_norm[dr_norm < eps] = eps

        dr /= dr_norm[..., np.newaxis]  # 转化为方向向量

        alpha = 3 * np.einsum('...i,...j->...ij', dr, dr) - np.identity(3)
        beta = 4 * pi * dr_norm[..., np.newaxis, np.newaxis] ** 3
        kernel = mu_0 / beta * alpha
        kernel = kernel.transpose(0, 2, 3, 1).reshape(n_obs, 3, n_dipoles * 3)

        n_cols = [n_dipoles * 3]
        if self.store_sensitivities == "forward_only":
            kernel = np.einsum('...i,i->...', kernel, self.moment)
            n_cols = []

        tmi_G = None
        ret = []

        for c in components:
            if c == 'tmi':
                if tmi_G is None:
                    tmi_G = np.einsum('ij...,j->i...', kernel, self.tmi_projection)
                ret.append(tmi_G)
            elif c == 'bx':
                ret.append(kernel[:, 0, ...])
            elif c == 'by':
                ret.append(kernel[:, 1, ...])
            elif c == 'bz':
                ret.append(kernel[:, 2, ...])
            else:
                raise RuntimeError(f'Unsupported component `{c}`.')

        return np.concatenate([r[:, np.newaxis, ...] for r in ret], axis=1).reshape(-1, *n_cols)

    @property
    def chi(self):
        # 只用作兼容Simulation3DIntegral行为
        return self.moment

    @property
    def chiMap(self):
        # 只用作兼容Simulation3DIntegral行为
        return self.momentMap

    @property
    def chiDeriv(self):
        # 只用作兼容Simulation3DIntegral行为
        return self.momentDeriv
