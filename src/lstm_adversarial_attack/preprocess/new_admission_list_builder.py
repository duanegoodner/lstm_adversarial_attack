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
class NewAdmissionListBuilder(pre.AbstractAdmissionListBuilder):
    def __init__(
        self,
        resources: pre.NewAdmissionListBuilderResources,
        output_dir: Path = cfp.FULL_ADMISSION_LIST_OUTPUT,
        settings: NewAdmissionListBuilderSettings = None,
    ):
        self.resources = resources
        self._output_dir = output_dir
        if settings is None:
            settings = NewAdmissionListBuilderSettings()
        self._settings = settings

    @property
    def settings(self) -> NewAdmissionListBuilderSettings:
        return self._settings

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @cached_property
    def _filtered_icustay_bg_lab_vital(self) -> pd.DataFrame:
        return self.resources.icustay_bg_lab_vital.drop(
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
            ) for item in list_of_group_dfs
        ]

    def process(self) -> pre.NewAdmissionListBuilderOutput:
        return pre.NewAdmissionListBuilderOutput(
            admission_list=self.admission_list
        )
