import numpy as np

from ..formatter_mixin import FormatterMixin


class BasePFDataFormatterMixin(FormatterMixin):
    def format_data(self, this, data):
        data_list = []
        index = 0
        for src in this.survey.source_field.receiver_list:
            count = len(src.components) * len(src.locations)
            end = index + count
            data_list.append(self.format_data_single(data[index:end], src.components))
            index = end

        if len(data_list) == 1:
            return data_list[0]
        else:
            if self.pandas:
                import pandas as pd
                return pd.concat(data_list)
            else:
                return data_list

    def format_data_single(self, this, data, components):
        data = self.format_numpy_data(data, components)
        if self.pandas:
            data = self.to_pandas_dataframe(data, components)

        if self.locations:
            loc = this.survey.receiver_locations
            if self.pandas:
                import pandas as pd
                n_dims = loc.shape[-1]
                loc = pd.DataFrame(loc, columns=['x', 'y', 'z'][:n_dims])
                data = pd.concat([loc, data], axis=1)
            else:
                data = np.c_[loc, data]

        return data

    def format_numpy_data(self, _, data, components):
        raise NotImplementedError()

    def to_pandas_dataframe(self, _, data, components):
        import pandas as pd
        return pd.DataFrame(data, columns=components)


class FormattedBasePFSimulationMixin(BasePFDataFormatterMixin):
    def format_numpy_data(self, this, data, components):
        return data.reshape(-1, len(components))
