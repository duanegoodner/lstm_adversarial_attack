import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfs
import lstm_adversarial_attack.preprocess.new_preprocessor as pre
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.preprocess.resource_data_structs as rds


@dataclass
class NewAdmissionListBuilderSettings:
    measurement_cols: list[str] = field(
        default_factory=lambda: cfs.PREPROCESS_BG_DATA_COLS
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
        resources: rds.NewAdmissionListBuilderResources = None,
        output_dir: Path = None,
        settings: NewAdmissionListBuilderSettings = None,
        output_constructors: rds.NewAdmissionListBuilderOutputConstructors = None,
    ):
        if resources is None:
            resources = rds.NewAdmissionListBuilderResources()
        if output_dir is None:
            output_dir = cfp.FULL_ADMISSION_LIST_OUTPUT
        if settings is None:
            settings = NewAdmissionListBuilderSettings()
        if output_constructors is None:
            output_constructors = rds.NewAdmissionListBuilderOutputConstructors()
        super().__init__(
            resources=resources,
            output_dir=output_dir,
            settings=settings,
            output_constructors=output_constructors,
        )
        self.icustay_bg_lab_vital = resources.icustay_bg_lab_vital.item

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
    def admission_list(self) -> list[eds.NewFullAdmissionData]:
        df_grouped_by_hadm = self._filtered_icustay_bg_lab_vital.groupby(
            ["hadm_id"]
        )
        list_of_group_dfs = [group[1] for group in df_grouped_by_hadm]
        return [
            eds.NewFullAdmissionData(
                subject_id=int(np.unique(item.subject_id)[0]),
                hadm_id=int(np.unique(item.hadm_id)[0]),
                icustay_id=int(np.unique(item.icustay_id)[0]),
                admittime=pd.Timestamp(np.unique(item.admittime)[0]),
                dischtime=pd.Timestamp(np.unique(item.dischtime)[0]),
                hospital_expire_flag=int(
                    np.unique(item.hospital_expire_flag)[0]
                ),
                intime=pd.Timestamp(np.unique(item.intime)[0]),
                outtime=pd.Timestamp(np.unique(item.outtime)[0]),
                time_series=item[self.settings.time_series_cols],
            )
            for item in list_of_group_dfs
        ]

    def process(self) -> dict[str, rds.OutgoingPreprocessResource]:
        return {
            "full_admission_list": self.output_constructors.full_admission_list(
                resource=self.admission_list
            )
        }


if __name__ == "__main__":
    init_start = time.time()
    full_admission_list_builder = NewAdmissionListBuilder()
    init_end = time.time()
    print(f"list builder init time = {init_end - init_start}")

    start_process = time.time()
    result = full_admission_list_builder.process()
    end_process = time.time()
    print(f"process time = {end_process - start_process}")

    start_json_export = time.time()
    for key, outgoing_resource in result.items():
        outgoing_resource.export(
            path=full_admission_list_builder.output_dir / f"{key}{outgoing_resource.file_ext}"
        )
    end_json_export = time.time()
    print(f"export time = {end_json_export - start_json_export}")
