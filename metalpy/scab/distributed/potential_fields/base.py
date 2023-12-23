from __future__ import annotations

import copy
import inspect
import os

import numpy as np
from SimPEG.potential_fields.base import BasePFSimulation

from metalpy.scab import simpeg_patched
from metalpy.utils.arg_specs import ArgSpecs
from ..base import BaseDistributedSimulationMixin


class DistributedBasePFSimulationMixin(BaseDistributedSimulationMixin):
    def linear_operator(this, self: BasePFSimulation):
        n_cells = self.nC
        if getattr(self, "model_type", None) == "vector":
            n_cells *= 3

        if self.store_sensitivities == "disk":
            sens_name = os.path.join(self.sensitivity_path, "sensitivity.npy")
            if os.path.exists(sens_name):
                # do not pull array completely into ram, just need to check the size
                kernel = np.load(sens_name, mmap_mode="r")
                if kernel.shape == (self.survey.nD, n_cells):
                    print(f"Found sensitivity file at {sens_name} with expected shape")
                    kernel = np.asarray(kernel)
                    return kernel

        store_sensitivities = self.store_sensitivities

        sim_cls = type(self)
        mesh = this.compress_mesh()
        kwargs = this.build_kwargs()
        surveys = this.divide_surveys()
        patches = this.distribute_patches()

        def worker(survey, patches):
            with simpeg_patched(*patches):
                simulation = sim_cls(
                    survey=survey,
                    mesh=mesh(),
                    **kwargs
                )
            return simulation.linear_operator()

        if not this.executor.shares_memory():
            result = this.executor.map(worker, surveys, patches)
            kernel = np.concatenate(result)
        else:
            if store_sensitivities == "forward_only":
                kernel_shape = (self.survey.nD,)
            else:
                kernel_shape = (self.survey.nD, n_cells)

            dtype = self.sensitivity_dtype
            kernel = this.executor.create_shared_array(kernel_shape, dtype=dtype)

            nDs = np.asarray([survey.nD for survey in surveys])
            i1s = nDs.cumsum()
            i0s = i1s - nDs

            def indexed_worker(survey, patches, i0, i1):
                ret = worker(survey, patches)
                kernel[i0:i1] = ret

            _ = this.executor.map(indexed_worker, surveys, patches, i0s, i1s)

        if self.store_sensitivities == "disk":
            print(f"writing sensitivity to {sens_name}")
            os.makedirs(self.sensitivity_path, exist_ok=True)
            np.save(sens_name, kernel)

        return kernel

    def divide_surveys(self, this: BasePFSimulation | None = None):
        survey = this.survey
        n_obs = survey.nD
        remaining_receivers = [copy.copy(p) for p in survey.source_field.receiver_list]

        workers = self.executor.get_workers()
        weights = np.asarray([w.get_weight() for w in workers])
        total_weight = sum(weights)

        counts = weights / total_weight * n_obs

        result_surveys = []
        rx = remaining_receivers.pop(0)
        for count in counts[:-1]:
            temp_receivers = []

            while rx.nD < count:
                temp_receivers.append(rx)
                count -= rx.nD
                rx = remaining_receivers.pop(0)

            n_component = len(rx.components)
            n_loc = max(int(count / n_component), 1)
            temp_receivers.append(type(rx)(rx.locations[:n_loc], rx.components))
            rx.locations = rx.locations[n_loc:]

            result_surveys.append(reassemble_survey(survey, temp_receivers))

        result_surveys.append(reassemble_survey(survey, [rx] + remaining_receivers))

        return result_surveys

    def build_kwargs(self, this: BasePFSimulation | None = None):
        sim_cls = type(this)
        spec = ArgSpecs.of(sim_cls)
        for cls in inspect.getmro(sim_cls)[1:]:
            spec.merge_kwargs(cls)
            if cls is BasePFSimulation:
                break

        kwargs = {}
        for key in spec.kwarg_specs | set(spec.posarg_specs):
            if key.name.endswith("Map"):  # 直接传递映射后的模型值，不传递mapping本身
                continue
            value = getattr(this, key.name, None)
            if value is not None:
                kwargs[key.name] = value

        kwargs.pop('mesh', None)  # mesh需要单独进行压缩操作
        kwargs.pop('survey', None)  # survey单独完成分割并传递

        return kwargs


def reassemble_survey(base_survey, new_receivers):
    survey = copy.copy(base_survey)
    survey.source_field = copy.copy(survey.source_field)

    survey.source_field.receiver_list = new_receivers

    return survey
