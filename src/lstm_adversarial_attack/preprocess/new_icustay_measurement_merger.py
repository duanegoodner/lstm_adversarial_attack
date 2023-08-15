import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfs
import lstm_adversarial_attack.preprocess.new_preprocessor as pre
import lstm_adversarial_attack.preprocess.resource_data_structs as rds


@dataclass
class NewICUStayMeasurementMergerSettings:
    """
    Container for ICUStayMeasurementCombiner config settings
    """

    bg_data_cols: list[str] = field(
        default_factory=lambda: cfs.PREPROCESS_BG_DATA_COLS
    )
    lab_data_cols: list[str] = field(
        default_factory=lambda: cfs.PREPROCESS_LAB_DATA_COLS
    )
    vital_data_cols: list[str] = field(
        default_factory=lambda: cfs.PREPROCESS_VITAL_DATA_COLS
    )

    @property
    def all_measurement_cols(self) -> list[str]:
        return self.bg_data_cols + self.lab_data_cols + self.vital_data_cols


class NewICUStayMeasurementMerger(pre.NewPreprocessModule):
    def __init__(
        self,
        resources: rds.NewICUStayMeasurementMergerResources = None,
        output_dir: Path = None,
        settings: NewICUStayMeasurementMergerSettings = None,
    ):
        if resources is None:
            resources = rds.NewICUStayMeasurementMergerResources()
        if output_dir is None:
            output_dir = cfp.STAY_MEASUREMENT_OUTPUT
        if settings is None:
            settings = NewICUStayMeasurementMergerSettings()
        super().__init__(
            resources=resources, output_dir=output_dir, settings=settings
        )
        self.prefiltered_icustay =resources.prefiltered_icustay.item
        self.prefiltered_bg = resources.prefiltered_bg.item
        self.prefiltered_vital = resources.prefiltered_vital.item
        self.prefiltered_lab = resources.prefiltered_lab.item

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

    def process(self) -> dict[str, rds.OutgoingPreprocessResource]:
        return {
            "icustay_bg_lab_vital": rds.OutgoingPreprocessDataFrame(
                resource=self.icustay_bg_lab_vital
            ),
            "bg_lab_vital_summary_stats": rds.OutgoingPreprocessDataFrame(
                resource=self.summary_stats
            ),
        }


if __name__ == "__main__":
    init_start = time.time()
    measurement_merger = NewICUStayMeasurementMerger()
    init_end = time.time()
    print(f"measurement merger init time = {init_end - init_start}")

    process_start = time.time()
    result = measurement_merger.process()
    process_end = time.time()
    print(f"measurement merger process time = {process_end - process_start}")

    export_start = time.time()
    for key, outgoing_dataframe in result.items():
        outgoing_dataframe.export(
            path=measurement_merger.output_dir / f"{key}.feather"
        )
    export_end = time.time()
    print(f"measurement merger export time = {export_end - export_start}")
