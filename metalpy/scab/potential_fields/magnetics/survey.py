from SimPEG.potential_fields import magnetics


class VectorizedSurvey(magnetics.Survey):
    """行为与原本的Survey相同，但是每次传递每个receiver组的所有观测点，方便实现了向量化的Simulation使用
    """
    def _location_component_iterator(self):
        for rx in self.source_field.receiver_list:
            yield rx.locations, rx.components
