import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import preprocess_module as pm
import preprocess_input_classes as pic
from lstm_adversarial_attack.config_paths import PREPROCESS_OUTPUT_FILES
# import project_config_old as pc


@dataclass
class FeatureFinalizerResources:
    processed_admission_list: list[pic.FullAdmissionData]


class FeatureFinalizer(pm.PreprocessModule):
    def __init__(
        self,
        settings=pic.FeatureFinalizerSettings(),
        incoming_resource_refs=pic.FeatureFinalizerResourceRefs(),
    ):
        super().__init__(
            settings=settings, incoming_resource_refs=incoming_resource_refs
        )

    def _import_resources(self) -> FeatureFinalizerResources:
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
            / PREPROCESS_OUTPUT_FILES["measurement_data_list"],
        )

        self.export_resource(
            key="in_hospital_mortality_list",
            resource=in_hospital_mortality_list,
            path=self.settings.output_dir
            / PREPROCESS_OUTPUT_FILES["in_hospital_mortality_list"],
        )


if __name__ == "__main__":
    feature_finalizer = FeatureFinalizer()
    exported_resources = feature_finalizer()
