import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfs
import lstm_adversarial_attack.preprocess.new_preprocessor as pre


@dataclass
class NewFeatureFinalizerSettings:
    """
    Container for FeatureFinalizer config settings
    """

    max_hours: int = cfs.MAX_OBSERVATION_HOURS
    min_hours: int = cfs.MIN_OBSERVATION_HOURS
    require_exact_num_hours: bool = (
        cfs.REQUIRE_EXACT_NUM_HOURS  # when True, no need for padding
    )
    observation_window_start: str = cfs.OBSERVATION_WINDOW_START


@dataclass
class NewFeatureFinalizer(pre.AbstractFeatureFinalizer):
    def __init__(
        self,
        resources: pre.NewFeatureFinalizerResources,
        output_dir: cfp.PREPROCESS_OUTPUT_DIR = cfp.FEATURE_FINALIZER_OUTPUT,
        settings: NewFeatureFinalizerSettings = None,
    ):
        self.resources = resources
        self._output_dir = output_dir
        if settings is None:
            settings = NewFeatureFinalizerSettings()
        self._settings = settings

    @property
    def settings(self) -> NewFeatureFinalizerSettings:
        return self._settings

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @cached_property
    def measurement_col_names(self) -> tuple[str, ...]:
        all_data_col_names = [
            item.time_series.columns
            for item in self.resources.processed_admission_list
        ]
        first_item_names = list(all_data_col_names[0])
        assert all(
            [(names == first_item_names).all() for names in all_data_col_names]
        )
        first_item_names.remove("charttime")
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
        self, sample: pre.NewFullAdmissionData
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
        measurement_data_list = []
        in_hospital_mortality_list = []

        for entry in self.resources.processed_admission_list:
            feature_array = self._get_feature_array(sample=entry)
            if feature_array is not None:
                measurement_data_list.append(feature_array)
                in_hospital_mortality_list.append(
                    entry.hospital_expire_flag.item()
                )

        return pre.NewFeatureFinalizerOutput(
            measurement_col_names=self.measurement_col_names,
            measurement_data_list=measurement_data_list,
            in_hospital_mortality_list=in_hospital_mortality_list,
        )
