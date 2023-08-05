from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import numpy as np
import pandas as pd
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfs
import lstm_adversarial_attack.preprocess.new_preprocessor as pre


@dataclass
class NewAdmissionListBuilderSettings:
    measurement_cols: list[str] = None

    def __post_init__(self):
        if self.measurement_cols is None:
            self.measurement_cols = (
                cfs.PREPROCESS_BG_DATA_COLS
                + cfs.PREPROCESS_LAB_DATA_COLS
                + cfs.PREPROCESS_VITAL_DATA_COLS
            )

    @property
    def time_series_cols(self) -> list[str]:
        return ["charttime"] + self.measurement_cols


@dataclass
class NewAdmissionListBuilder(pre.NewPreprocessModule):
    def __init__(
        self,
        resources: dict[str, pre.IncomingFeatherDataFrame] = None,
        output_dir: Path = cfp.FULL_ADMISSION_LIST_OUTPUT,
        settings: NewAdmissionListBuilderSettings = None,
    ):
        if resources is None:
            resources = {
                "icustay_bg_lab_vital": pre.IncomingFeatherDataFrame(
                    resource_id=cfp.FULL_ADMISSION_LIST_INPUT_FILES[
                        "icustay_bg_lab_vital"
                    ]
                )
            }
        if settings is None:
            settings = NewAdmissionListBuilderSettings()
        super().__init__(
            resources=resources, output_dir=output_dir, settings=settings
        )
        self.icustay_bg_lab_vital = self.resource_items["icustay_bg_lab_vital"]

    @cached_property
    def _filtered_icustay_bg_lab_vital(self) -> pd.DataFrame:
        return self.icustay_bg_lab_vital.drop(
            labels=[
                "dod",
                "los_hospital",
                "admission_age",
                "hospstay_seq",
                "icustay_seq",
                "first_hosp_stay",
                "los_icu",
                "first_icu_stay",
            ],
            axis=1,
        )

    @cached_property
    def admission_list(self) -> list[pre.NewFullAdmissionData]:
        df_grouped_by_hadm = self._filtered_icustay_bg_lab_vital.groupby(
            ["hadm_id"]
        )
        list_of_group_dfs = [group[1] for group in df_grouped_by_hadm]
        return [
            pre.NewFullAdmissionData(
                subject_id=np.unique(item.subject_id),
                hadm_id=np.unique(item.hadm_id),
                icustay_id=np.unique(item.icustay_id),
                admittime=np.unique(item.admittime),
                dischtime=np.unique(item.dischtime),
                hospital_expire_flag=np.unique(item.hospital_expire_flag),
                intime=np.unique(item.intime),
                outtime=np.unique(item.outtime),
                time_series=item[self.settings.time_series_cols],
            )
            for item in list_of_group_dfs
        ]

    def process(self) -> dict[str, pre.OutgoingPreprocessResource]:
        return {
            "full_admission_list": pre.OutgoingPreprocessResourceNoExport(
                resource=self.admission_list
            )
        }


if __name__ == "__main__":
    full_admission_list_builder = NewAdmissionListBuilder()
    result = full_admission_list_builder.process()
