import pandas as pd
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.preprocess.preprocess_module as pm
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic


@dataclass
class ICUStayMeasurementCombinerResources:
    icustay: pd.DataFrame
    bg: pd.DataFrame
    lab: pd.DataFrame
    vital: pd.DataFrame


class ICUStayMeasurementCombiner(pm.PreprocessModule):
    def __init__(
        self,
        settings=pic.ICUStayMeasurementCombinerSettings(),
        incoming_resource_refs=pic.ICUStayMeasurementCombinerResourceRefs(),
    ):
        super().__init__(
            name="ICU Stay Data + Measurement Data Combiner",
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
        )

    def _import_resources(self) -> ICUStayMeasurementCombinerResources:
        imported_data = ICUStayMeasurementCombinerResources(
            icustay=self.import_pickle_to_df(
                self.incoming_resource_refs.icustay
            ),
            bg=self.import_pickle_to_df(self.incoming_resource_refs.bg),
            lab=self.import_pickle_to_df(self.incoming_resource_refs.lab),
            vital=self.import_pickle_to_df(self.incoming_resource_refs.vital),
        )

        return imported_data

    def create_id_bg(
        self, bg: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.merge(
            left=icustay,
            right=bg,
            on=["hadm_id"],
            how="right",
            suffixes=("_icu", "_bg"),
        )[
            [
                "subject_id",
                "hadm_id",
                "icustay_id_icu",
                "charttime",
            ]
            + self.settings.bg_data_cols
        ].rename(
            columns={"icustay_id_icu": "icustay_id"}
        )

    def create_id_lab(
        self, lab: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        return (
            pd.merge(
                left=icustay,
                right=lab,
                on=["hadm_id"],
                how="right",
                suffixes=("_icu", "_lab"),
            )
            .rename(columns={"subject_id_icu": "subject_id"})[
                [
                    "subject_id",
                    "hadm_id",
                    "icustay_id_icu",
                    "charttime",
                ]
                + self.settings.lab_data_cols
            ]
            .rename(columns={"icustay_id_icu": "icustay_id"})
        )

    # @staticmethod
    def create_id_vital(
        self, vital: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.merge(left=icustay, right=vital, on=["icustay_id"])[
            [
                "subject_id",
                "hadm_id",
                "icustay_id",
                "charttime",
            ]
            + self.settings.vital_data_cols
        ]

    @staticmethod
    def merge_measurement_sources(
        id_bg: pd.DataFrame,
        id_lab: pd.DataFrame,
        id_vital: pd.DataFrame,
    ):
        id_bg_lab = pd.merge(
            left=id_bg,
            right=id_lab,
            on=["subject_id", "hadm_id", "icustay_id", "charttime"],
            how="outer",
        )
        id_bg_lab_vital = pd.merge(
            left=id_bg_lab,
            right=id_vital,
            on=["subject_id", "hadm_id", "icustay_id", "charttime"],
            how="outer",
            suffixes=("_bglab", "_vital"),
        )

        return id_bg_lab_vital

    @staticmethod
    def combine_icustay_info_with_measurements(
        id_bg_lab_vital: pd.DataFrame,
        icustay: pd.DataFrame,
    ) -> pd.DataFrame:
        info_id_bg_lab_vital = pd.merge(
            left=icustay,
            right=id_bg_lab_vital,
            on=["subject_id", "hadm_id", "icustay_id"],
            how="outer",
        )

        return info_id_bg_lab_vital

    def process(self):
        data = self._import_resources()
        id_bg = self.create_id_bg(bg=data.bg, icustay=data.icustay)
        id_lab = self.create_id_lab(lab=data.lab, icustay=data.icustay)
        id_vital = self.create_id_vital(vital=data.vital, icustay=data.icustay)
        id_bg_lab_vital = self.merge_measurement_sources(
            id_bg=id_bg, id_lab=id_lab, id_vital=id_vital
        )
        icustay_bg_lab_vital = self.combine_icustay_info_with_measurements(
            id_bg_lab_vital=id_bg_lab_vital, icustay=data.icustay
        )

        summary_stats = icustay_bg_lab_vital[
            self.settings.all_measurement_cols
        ].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])

        self.export_resource(
            key="icustay_bg_lab_vital",
            resource=icustay_bg_lab_vital,
            path=self.settings.output_dir
            / lcp.STAY_MEASUREMENT_OUTPUT_FILES["icustay_bg_lab_vital"],
        )

        self.export_resource(
            key="bg_lab_vital_summary_stats",
            resource=summary_stats,
            path=self.settings.output_dir
            / lcp.STAY_MEASUREMENT_OUTPUT_FILES["bg_lab_vital_summary_stats"],
        )


if __name__ == "__main__":
    icustay_measurement_combiner = ICUStayMeasurementCombiner()
    exported_resources = icustay_measurement_combiner()
