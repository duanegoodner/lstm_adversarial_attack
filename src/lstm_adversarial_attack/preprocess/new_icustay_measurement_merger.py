from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import pandas as pd
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfs
import lstm_adversarial_attack.preprocess.new_preprocessor as pre


@dataclass
class NewICUStayMeasurementMergerSettings:
    """
    Container for ICUStayMeasurementCombiner config settings
    """

    bg_data_cols: list[str] = None
    lab_data_cols: list[str] = None
    vital_data_cols: list[str] = None

    def __post_init__(self):
        if self.bg_data_cols is None:
            self.bg_data_cols = cfs.PREPROCESS_BG_DATA_COLS
        if self.lab_data_cols is None:
            self.lab_data_cols = cfs.PREPROCESS_LAB_DATA_COLS
        if self.vital_data_cols is None:
            self.vital_data_cols = cfs.PREPROCESS_VITAL_DATA_COLS

    @property
    def all_measurement_cols(self) -> list[str]:
        return self.bg_data_cols + self.lab_data_cols + self.vital_data_cols


class NewICUStayMeasurementMerger(pre.NewPreprocessModule):
    def __init__(
        self,
        resources: dict[str, pre.IncomingFeatherDataFrame] = None,
        output_dir: Path = cfp.STAY_MEASUREMENT_OUTPUT,
        settings: NewICUStayMeasurementMergerSettings = None,
    ):
        if resources is None:
            resources = {
               key: pre.IncomingFeatherDataFrame(resource_id=value) for key, value
               in cfp.STAY_MEASUREMENT_INPUT_FILES.items()
            }
        if settings is None:
            settings = NewICUStayMeasurementMergerSettings()
        super().__init__(
            resources=resources, output_dir=output_dir, settings=settings
        )
        self.prefiltered_icustay = self.resource_items["prefiltered_icustay"]
        self.prefiltered_bg = self.resource_items["prefiltered_bg"]
        self.prefiltered_vital = self.resource_items["prefiltered_vital"]
        self.prefiltered_lab = self.resource_items["prefiltered_lab"]

    @cached_property
    def _id_bg(self) -> pd.DataFrame:
        return pd.merge(
            left=self.prefiltered_icustay,
            right=self.prefiltered_bg,
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

    @cached_property
    def _id_lab(self) -> pd.DataFrame:
        return (
            pd.merge(
                left=self.prefiltered_icustay,
                right=self.prefiltered_lab,
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

    @cached_property
    def _id_vital(self) -> pd.DataFrame:
        return pd.merge(
            left=self.prefiltered_icustay,
            right=self.prefiltered_vital,
            on=["icustay_id"],
        )[
            [
                "subject_id",
                "hadm_id",
                "icustay_id",
                "charttime",
            ]
            + self.settings.vital_data_cols
        ]

    @cached_property
    def _id_bg_lab_vital(self) -> pd.DataFrame:
        id_bg_lab = pd.merge(
            left=self._id_bg,
            right=self._id_lab,
            on=["subject_id", "hadm_id", "icustay_id", "charttime"],
            how="outer",
        )
        return pd.merge(
            left=id_bg_lab,
            right=self._id_vital,
            on=["subject_id", "hadm_id", "icustay_id", "charttime"],
            how="outer",
            suffixes=("_bglab", "_vital"),
        )

    @cached_property
    def icustay_bg_lab_vital(self) -> pd.DataFrame:
        merged_df = pd.merge(
            left=self.prefiltered_icustay,
            right=self._id_bg_lab_vital,
            on=["subject_id", "hadm_id", "icustay_id"],
            how="outer",
        )
        return merged_df

    @cached_property
    def summary_stats(self) -> pd.DataFrame:
        summary = self.icustay_bg_lab_vital[
            self.settings.all_measurement_cols
        ].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
        return summary

    def process(self) -> dict[str, pre.OutgoingPreprocessResource]:
        return {
            "icustay_bg_lab_vital": pre.OutgoingPreprocessDataFrame(
                resource=self.icustay_bg_lab_vital
            ),
            "bg_lab_vital_summary_stats": pre.OutgoingPreprocessDataFrame(
                resource=self.summary_stats
            )
        }

        # return pre.NewICUStayMeasurementMergerOutput(
        #     icustay_bg_lab_vital=self.icustay_bg_lab_vital,
        #     bg_lab_vital_summary_stats=self.summary_stats,
        # )

if __name__ == "__main__":
    measurement_merger = NewICUStayMeasurementMerger()
    result = measurement_merger.process()
    for key, outgoing_dataframe in result.items():
        outgoing_dataframe.export(
            path=cfp.STAY_MEASUREMENT_OUTPUT / f"{key}.feather"
        )