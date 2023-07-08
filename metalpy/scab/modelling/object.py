from __future__ import annotations

from typing import Any, Union

from metalpy.utils.dhash import dhash
from metalpy.utils.model import pv_ufunc_assign, DataAssociation
from .mix_modes import MixMode, Mixer, dhashable_mixer
from .shapes import Shape3D


class Object:
    DEFAULT_KEY = '__DEFAULT__'

    def __init__(self,
                 shape: Shape3D,
                 models: Union[dict[str, Any], Any] | None = None,
                 mix_mode: Mixer = MixMode.Override):
        """代表一个三维几何体图层

        Parameters
        ----------
        shape
            三维几何体
        models
            三维几何体的参数
        mix_mode
            混合模式，必须为MixMode枚举或自定义的混合函数，函数的参数见 Object.mix
        """
        self.shape = shape
        if models is None:
            models = True
        if not isinstance(models, dict):
            models = {Object.DEFAULT_KEY: models}
        self.models = models

        self.mix_mode = mix_mode

    def to_polydata(self):
        shape = self.shape
        poly = shape.to_polydata()

        if poly is not None:
            for k, v in self.models.items():
                pv_ufunc_assign(poly, DataAssociation.Cell, k, v, inplace=True)

        return poly

    @property
    def mixer(self):
        return MixMode.dispatch(self.mix_mode)

    @property
    def shape(self) -> Shape3D:
        return self._shape

    @shape.setter
    def shape(self, val: Shape3D):
        self._shape = val

    @property
    def models(self) -> dict[str, Any]:
        return self._models

    @models.setter
    def models(self, val: dict[str, Any]):
        self._models = val

    @property
    def model(self) -> Any:
        return self.models[Object.DEFAULT_KEY]

    @model.setter
    def model(self, val: Any):
        self.models[Object.DEFAULT_KEY] = val

    @property
    def n_tasks(self):
        return self.shape.n_tasks

    @property
    def progress_manually(self):
        return self.shape.progress_manually

    def items(self):
        for k, v in self.models.items():
            yield k, v

    def mix(self, prev_layer, current_layer):
        """执行模型混合操作

        Parameters
        ----------
        prev_layer
            之前所有模型结果和当前模型的重合部分的值数组
        current_layer
            当前模型重合部分的值数据

        Returns
        -------
        ret
            混合操作后的结果
        """
        return self.mixer(prev_layer, current_layer)

    def __getitem__(self, item):
        return self.models[item]

    def __dhash__(self):
        return dhash(self.shape, dhashable_mixer(self.mix_mode), self.models)
