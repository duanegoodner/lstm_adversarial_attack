import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.preprocess.preprocess_module as pm
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic
import lstm_adversarial_attack.config_paths as lcp


@dataclass
class FullAdmissionListBuilderResources:
    icustay_bg_lab_vital: pd.DataFrame


class FullAdmissionListBuilder(pm.PreprocessModule):
    def __init__(
        self,
        settings=pic.FullAdmissionListBuilderSettings(),
        incoming_resource_refs=pic.FullAdmissionListBuilderResourceRefs(),
    ):
        super().__init__(
            name="FullAdmission Object List Builder",
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
        )

    def _import_resources(self) -> FullAdmissionListBuilderResources:
        imported_data = FullAdmissionListBuilderResources(
            icustay_bg_lab_vital=self.import_pickle_to_df(
                path=self.incoming_resource_refs.icustay_bg_lab_vital
            )
        )
        return imported_data

    def process(self):
        data = self._import_resources()
        data.icustay_bg_lab_vital = data.icustay_bg_lab_vital.drop(
            [
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

        df_grouped_by_hadm = data.icustay_bg_lab_vital.groupby(["hadm_id"])
        list_of_group_dfs = [group[1] for group in df_grouped_by_hadm]
        full_admission_list = [
            pic.FullAdmissionData(
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

        output_path = (
            self.settings.output_dir
            / lcp.FULL_ADMISSION_LIST_OUTPUT_FILES["full_admission_list"]
        )
        self.export_resource(
            key="full_admission_list",
            resource=full_admission_list,
            path=output_path,
        )


if __name__ == "__main__":
    full_admission_list_builder = FullAdmissionListBuilder()
    exported_resources = full_admission_list_builder()
