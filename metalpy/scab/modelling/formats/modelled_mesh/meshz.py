from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from discretize import TensorMesh

from metalpy.utils.file import PathLike

if TYPE_CHECKING:
    from typing import TypeVar
    from metalpy.scab.modelling.modelled_mesh import ModelledMesh
    TMesh = TypeVar('TMesh', bound=ModelledMesh)


class MeshZFormat:
    Origin = 'o'  # 网格原点关键字
    H = ['hx', 'hy', 'hz']  # 网格尺寸关键字
    Hc = ['hcx', 'hcy', 'hcz']  # 压缩存储的网格数关键字

    DefaultKey = 'default'  # 默认模型关键字
    IndActive = 'active'  # 有效网格索引关键字
    ModelPrefix = '@model:'  # 模型存储关键字前缀

    def to_meshz(
        self: 'TMesh',
        path: PathLike,
    ):
        path = Path(path)
        mesh = self.mesh
        assert isinstance(mesh, TensorMesh), 'Only `TensorMesh` is supported for now.'  # TODO: 支持 TreeMesh

        data = {}

        hs = self.h
        for i in range(len(hs)):
            key = MeshZFormat.H[i]
            h = hs[i]
            splits = np.where(np.diff(h))[0]

            h_size = []
            h_count = []
            for x, y in np.lib.stride_tricks.sliding_window_view(np.r_[0, splits, len(h)], 2):
                h_size.append(h[x])
                h_count.append(y - x)

                if len(h_count) * 2 >= len(h):
                    # 压缩后长度大于原长度，不压缩
                    break
            else:
                # 正常结束循环，即压缩后长度小于原长度，使用压缩后的尺寸定义
                data[MeshZFormat.Hc[i]] = h_count
                h = h_size

            data[key] = h

        ind_active = self._ind_active
        should_check_ind_active = not self.is_active_indices
        ind_active_saved = False
        for key in self:
            model = self.get_raw_model(key)
            saved_key = MeshZFormat.ModelPrefix + key
            data[saved_key] = model

            if should_check_ind_active:
                if (
                        np.size(model) == np.size(ind_active)
                        and np.result_type(model, ind_active) is ind_active.dtype
                        and np.array_equal(model, ind_active)
                ):
                    data[MeshZFormat.IndActive] = key
                    should_check_ind_active = False
                    ind_active_saved = True

        data[MeshZFormat.Origin] = self.origin

        if not ind_active_saved:
            data[MeshZFormat.IndActive] = self._ind_active

        if self.has_default_model:
            data[MeshZFormat.DefaultKey] = self.default_key

        np.savez_compressed(path, **data)
        if path.suffix != '.npz':  # savez中如果文件后缀不是npz，会强制增加npz
            if path.exists():
                path.unlink()
            os.rename(path.with_suffix(path.suffix + '.npz'), path)

    @classmethod
    def from_meshz(
        cls: type['TMesh'],
        path: PathLike,
    ) -> 'TMesh':
        data = np.load(path)

        origin = data[MeshZFormat.Origin]
        h = []
        for i in range(len(MeshZFormat.H)):
            h_key = MeshZFormat.H[i]
            hc_key = MeshZFormat.Hc[i]
            if h_key not in data:
                break

            hs = data[h_key]

            if hc_key in data:
                # 检测到压缩存储的网格数参数，应为压缩存储格式
                # 构造网格规格
                hc = data[hc_key]
                hs = [(s, c) for s, c in zip(hs, hc)]

            h.append(hs)

        models = {}
        for key in data:
            if key.startswith(MeshZFormat.ModelPrefix):
                models[key[len(MeshZFormat.ModelPrefix):]] = data[key]

        ind_active = data[MeshZFormat.IndActive]
        if np.issubdtype(ind_active.dtype, str):
            ind_active = models[str(ind_active)]

        mesh = TensorMesh(h, origin=origin)

        ret = cls(mesh=mesh, models=models, ind_active=ind_active)

        if MeshZFormat.DefaultKey in data:
            ret.default_key = str(data[MeshZFormat.DefaultKey])

        return ret
