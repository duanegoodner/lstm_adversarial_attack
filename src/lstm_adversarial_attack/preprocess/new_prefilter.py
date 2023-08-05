import pandas as pd
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_set
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic
import lstm_adversarial_attack.preprocess.new_preprocessor as pre


@dataclass
class NewPrefilterSettings:
    """
    Container for objects imported by Prefilter
    """

    # output_dir: Path = cfg_paths.PREFILTER_OUTPUT
    min_age: int = 18
    min_los_hospital: int = 1
    min_los_icu: int = 1
    bg_data_cols: list[str] = None
    lab_data_cols: list[str] = None
    vital_data_cols: list[str] = None

    def __post_init__(self):
        if self.bg_data_cols is None:
            self.bg_data_cols = cfg_set.PREPROCESS_BG_DATA_COLS
        if self.lab_data_cols is None:
            self.lab_data_cols = cfg_set.PREPROCESS_LAB_DATA_COLS
        if self.vital_data_cols is None:
            self.vital_data_cols = cfg_set.PREPROCESS_VITAL_DATA_COLS


class NewPrefilter(pre.NewPreprocessModule):
    def __init__(
        self,
        resources: dict[str, pre.IncomingCSVDataFrame] = None,
        output_dir=cfg_paths.PREFILTER_OUTPUT,
        settings: NewPrefilterSettings = None,
    ):
        if resources is None:
            resources = {
                key: pre.IncomingCSVDataFrame(resource_id=value)
                for key, value in cfg_paths.PREFILTER_INPUT_FILES.items()
            }
        if settings is None:
            settings = NewPrefilterSettings()
        super().__init__(
            resources=resources, output_dir=output_dir, settings=settings
        )
        self.icustay = self.resource_items["icustay"]
        self.bg = self.resource_items["bg"]
        self.vital = self.resource_items["vital"]
        self.lab = self.resource_items["lab"]

    def _apply_standard_formatting(self):
        """
        Sets col names of all dfs in self.resouces to lowercase
        """

        module_dfs = [self.icustay, self.bg, self.vital, self.lab]

        for df in module_dfs:
            df.columns = [item.lower() for item in df.columns]

    def _filter_icustay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes certain data from icustay df:
        - patients below min age
        - stays where length of hosp stay or icu stay below min time

        Cutoff age and durations are set in:
        - lstm_adversarial_attack.config_settings

        :param df: icustay dataframe to be filtered
        :return: filtered dataframe
        """
        df["admittime"] = pd.to_datetime(df["admittime"])
        df["dischtime"] = pd.to_datetime(df["dischtime"])
        df["intime"] = pd.to_datetime(df["intime"])
        df["outtime"] = pd.to_datetime(df["outtime"])

        df = df[
            (df["admission_age"] >= self._settings.min_age)
            & (df["los_hospital"] >= self._settings.min_los_hospital)
            & (df["los_icu"] >= self._settings.min_los_icu)
        ]

        df = df.drop(["ethnicity", "ethnicity_grouped", "gender"], axis=1)

        return df

    @staticmethod
    def _filter_measurement_df(
        df: pd.DataFrame,
        identifier_cols: list[str],
        measurements_of_interest: list[str],
    ):
        """
        Filters a dataframe containing measurement data (lab, bg or vital).
        Called by other methods for each specific dataframe
        :param df: dataframe to be filtered
        :param identifier_cols: non-measurement metadata cols to keep
        :param measurements_of_interest: measurement vals to keep
        :return: filtered dataframe
        """
        df = df[identifier_cols + measurements_of_interest]
        df = df.dropna(subset=measurements_of_interest, how="all")
        return df

    def _filter_bg(
        self, bg: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filters the dataframe obtained from importing output of pivoted_bg.sql
        :param bg: dataframe with "blood group" data
        :param icustay: icustay dataframe. must already be filtered.
        :return: filtered bg dataframe
        """
        bg["charttime"] = pd.to_datetime(bg["charttime"])
        bg["icustay_id"] = bg["icustay_id"].fillna(0).astype("int64")
        bg = bg[bg["hadm_id"].isin(icustay["hadm_id"])]
        bg = self._filter_measurement_df(
            df=bg,
            identifier_cols=["icustay_id", "hadm_id", "charttime"],
            measurements_of_interest=self._settings.bg_data_cols,
        )

        return bg

    def _filter_lab(
        self, lab: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filters the dataframe obtained from importing output of pivoted_lab.sql
        :param lab: dataframe with "blood group" data
        :param icustay: icustay dataframe. must already be filtered.
        :return: filtered lab dataframe
        """
        lab["icustay_id"] = lab["icustay_id"].fillna(0).astype("int64")
        lab["hadm_id"] = lab["hadm_id"].fillna(0).astype("int64")
        lab["charttime"] = pd.to_datetime(lab["charttime"])
        lab = lab[lab["hadm_id"].isin(icustay["hadm_id"])]
        lab = self._filter_measurement_df(
            df=lab,
            identifier_cols=[
                "icustay_id",
                "hadm_id",
                "subject_id",
                "charttime",
            ],
            measurements_of_interest=self._settings.lab_data_cols,
        )
        return lab

    def _filter_vital(
        self, vital: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filters the dataframe obtained by importing output of pivoted_vital.sql
        :param vital: dataframe with "blood group" data
        :param icustay: icustay dataframe. must already be filtered.
        :return: filtered vital dataframe
        """
        vital["charttime"] = pd.to_datetime(vital["charttime"])
        vital = vital[vital["icustay_id"].isin(icustay["icustay_id"])]

        vital = self._filter_measurement_df(
            df=vital,
            identifier_cols=["icustay_id", "charttime"],
            measurements_of_interest=self._settings.vital_data_cols,
        )

        return vital

    def process(
        self,
    ) -> dict[str, pre.OutgoingPreprocessResource]:
        """
        Runs all filter methods.
        """
        self._apply_standard_formatting()
        filtered_icustay = self._filter_icustay(df=self.icustay)
        filtered_bg = self._filter_bg(bg=self.bg, icustay=filtered_icustay)
        filtered_lab = self._filter_lab(lab=self.lab, icustay=filtered_icustay)
        filtered_vital = self._filter_vital(
            vital=self.vital, icustay=filtered_icustay
        )

        return {
            "prefiltered_icustay": pre.OutgoingPreprocessDataFrame(
                resource=filtered_icustay
            ),
            "prefiltered_bg": pre.OutgoingPreprocessDataFrame(
                resource=filtered_bg
            ),
            "prefiltered_lab": pre.OutgoingPreprocessDataFrame(
                resource=filtered_lab
            ),
            "prefiltered_vital": pre.OutgoingPreprocessDataFrame(
                resource=filtered_vital
            ),
        }


if __name__ == "__main__":
    prefilter = NewPrefilter()
    result = prefilter.process()
    for key, outgoing_dataframe in result.items():
        outgoing_dataframe.export(
            path=cfg_paths.PREFILTER_OUTPUT / f"{key}.feather"
        )
