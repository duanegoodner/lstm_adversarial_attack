import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.preprocess.preprocess_module as pm
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic


@dataclass
class FeatureFinalizerResources:
    """
    Container for the data objects imported by FeatureFinalizer
    """
    processed_admission_list: list[pic.FullAdmissionData]


class FeatureFinalizer(pm.PreprocessModule):
    """
    Selects time range of data from FullAdmissionData from FeatureBuilder.

    Converts df features timeseries to numpy arrays

    Exports features, labels, and measurement names
    """
    def __init__(
        self,
        # settings=pic.FeatureFinalizerSettings(),
        # incoming_resource_refs=pic.FeatureFinalizerResourceRefs(),
    ):
        """
        Instantiates settings and resource references and passes to base class
        constructor
        """
        super().__init__(
            name="Feature Finalizer",
            settings=pic.FeatureFinalizerSettings(),
            incoming_resource_refs=pic.FeatureFinalizerResourceRefs(),
        )

    def _import_resources(self) -> FeatureFinalizerResources:
        """
        imports data from pickle file to container dataclass
        :return: dataclass with refs to imported data
        """
        imported_data = FeatureFinalizerResources(
            processed_admission_list=self.import_pickle_to_list(
                self.incoming_resource_refs.processed_admission_list
            )
        )
        return imported_data

    @staticmethod
    def _get_measurement_col_names(
        admission_list: list[pic.FullAdmissionData],
    ) -> tuple:
        """
        Gets names of measurement data cols & stores as tuple
        :param admission_list: list of FullAdmissionData
        :return: measurement column names
        """
        # confirm each sample's dataframe has same col names
        all_data_col_names = [
            item.time_series.columns for item in admission_list
        ]
        first_item_names = list(all_data_col_names[0])
        assert all(
            [(names == first_item_names).all() for names in all_data_col_names]
        )
        first_item_names.remove("charttime")

        # return as tuple (for fixed order)
        return tuple(first_item_names)

    @staticmethod
    def _get_features_in_time_range(
        time_series: pd.DataFrame,
        start_time: np.datetime64 = None,
        end_time: np.datetime64 = None,
        time_col: str = "charttime",
    ) -> pd.DataFrame:
        """
        Filters time series dataframe down to time window of interest
        :param time_series: df with measurements time series
        :param start_time: begin time of window (default to hosp admit time)
        :param end_time: end time of window
        :param time_col: timestamp col of incoming df
        :return: df with times filtered down to window of interest
        """
        if start_time is None:
            start_time = np.array([np.datetime64(datetime.min)])
        if end_time is None:
            end_time = np.array([np.datetime64(datetime.max)])
        return time_series[
            (time_series[time_col] > start_time[0])
            & (time_series[time_col] < end_time[0])
        ]

    def _get_feature_array(
        self, sample: pic.FullAdmissionData
    ) -> np.ndarray | None:
        """
        Filters time series df to window of interest & converts to array
        :param sample: FullAdmissionData object (has time series df attribute)
        :return: filtered numpy array of time series data
        """
        observation_start_time = getattr(
            sample, self.settings.observation_window_start
        )
        if self.settings.max_hours is not None:
            observation_end_time = observation_start_time + pd.Timedelta(
                hours=self.settings.max_hours
            )
        else:
            observation_end_time = None

        data_in_range = self._get_features_in_time_range(
            start_time=observation_start_time,
            end_time=observation_end_time,
            time_series=sample.time_series,
        )

        if data_in_range.shape[0] >= self.settings.min_hours:
            return data_in_range.loc[
                :, ~data_in_range.columns.isin(["charttime"])
            ].values
        else:
            return None

    def process(self):
        """
        Imports and filters time series.

        Exports list of time series arrays, list of labels, & col names.
        """
        assert self.settings.output_dir.exists()
        data = self._import_resources()
        measurement_col_names = self._get_measurement_col_names(
            data.processed_admission_list
        )

        measurement_data_list = []
        in_hospital_mortality_list = []

        for entry in data.processed_admission_list:
            feature_array = self._get_feature_array(sample=entry)
            if feature_array is not None:
                measurement_data_list.append(feature_array)
                in_hospital_mortality_list.append(
                    entry.hospital_expire_flag.item()
                )

        self.export_resource(
            key="measurement_col_names",
            resource=measurement_col_names,
            path=self.settings.output_dir / "measurement_col_names.pickle",
        )

        self.export_resource(
            key="measurement_data_list",
            resource=measurement_data_list,
            path=self.settings.output_dir
            / cfg_paths.PREPROCESS_OUTPUT_FILES["measurement_data_list"],
        )

        self.export_resource(
            key="in_hospital_mortality_list",
            resource=in_hospital_mortality_list,
            path=self.settings.output_dir
            / cfg_paths.PREPROCESS_OUTPUT_FILES["in_hospital_mortality_list"],
        )


if __name__ == "__main__":
    feature_finalizer = FeatureFinalizer()
    exported_resources = feature_finalizer()
