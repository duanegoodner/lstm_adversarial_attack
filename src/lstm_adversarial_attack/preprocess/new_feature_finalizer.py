import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property

import numpy as np
import pandas as pd

import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfs
import lstm_adversarial_attack.preprocess.new_preprocessor as pre
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.preprocess.resource_data_structs as rds


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
class NewFeatureFinalizer(pre.NewPreprocessModule):
    def __init__(
        self,
        resources: rds.NewFeatureFinalizerResources = None,
        output_dir: cfp.PREPROCESS_OUTPUT_DIR = None,
        settings: NewFeatureFinalizerSettings = None,
        output_info: rds.NewFeatureFinalizerOutputInfo = None,
    ):
        if resources is None:
            resources = rds.NewFeatureFinalizerResources()
        if output_dir is None:
            output_dir = cfp.FEATURE_FINALIZER_OUTPUT
        if settings is None:
            settings = NewFeatureFinalizerSettings()
        if output_info is None:
            output_info = rds.NewFeatureFinalizerOutputInfo()
        super().__init__(
            resources=resources,
            output_dir=output_dir,
            settings=settings,
            output_info=output_info,
        )
        self.processed_admission_list = resources.processed_admission_list.item

    @cached_property
    def measurement_col_names(self) -> tuple[str, ...]:
        all_data_col_names = [
            item.time_series.columns for item in self.processed_admission_list
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
            start_time = np.datetime64(datetime.min)
        if end_time is None:
            end_time = np.datetime64(datetime.max)
        return time_series[
            (time_series[time_col] > start_time)
            & (time_series[time_col] < end_time)
        ]

    def _get_feature_array(
        self, sample: eds.NewFullAdmissionData
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

    def process(
        self,
    ) -> dict[str, rds.JsonReadyOutput | rds.OutgoingFeaturesList]:
        measurement_data_list = []
        in_hospital_mortality_list = []

        for entry in self.processed_admission_list:
            feature_array = self._get_feature_array(sample=entry)
            if feature_array is not None:
                measurement_data_list.append(feature_array)
                in_hospital_mortality_list.append(entry.hospital_expire_flag)

        return {
            "in_hospital_mortality_list": (
                self.output_info.in_hospital_mortality_list(
                    resource=in_hospital_mortality_list
                )
            ),
            "measurement_col_names": self.output_info.measurement_col_names(
                resource=self.measurement_col_names,
            ),
            "measurement_data_list": self.output_info.measurement_data_list(
                resource=measurement_data_list
            ),
        }


if __name__ == "__main__":
    init_start = time.time()
    feature_finalizer = NewFeatureFinalizer()
    init_end = time.time()
    print(f"feature finalizer init time = {init_end - init_start}")

    process_start = time.time()
    result = feature_finalizer.process()
    process_end = time.time()
    print(f"process time = {process_end - process_start}")

    export_start = time.time()
    for key, outgoing_resource in result.items():
        outgoing_resource.export(
            path=feature_finalizer.output_dir
            / f"{key}{outgoing_resource.file_ext}"
        )
    export_end = time.time()
    print(f"export time = {export_end - export_start}")
