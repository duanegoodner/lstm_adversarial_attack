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


class NewPrefilter(pre.AbstractPrefilter):
    def __init__(
        self,
        resources: pre.NewPrefilterResources,
        output_dir = cfg_paths.PREFILTER_OUTPUT,
        settings: NewPrefilterSettings = None,
    ):
        self.resources = resources
        self._output_dir = output_dir
        if settings is None:
            settings = NewPrefilterSettings()
        self._settings = settings

    @property
    def settings(self) -> NewPrefilterSettings:
        return self._settings

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def _apply_standard_formatting(self):
        """
        Sets col names of all dfs in self.resouces to lowercase
        """
        for resource in self.resources.__dict__.values():
            if isinstance(resource, pd.DataFrame):
                resource.columns = [item.lower() for item in resource.columns]

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

    def process(self, ) -> pre.NewPrefilterOutput:
        """
        Runs all filter methods.
        """
        self._apply_standard_formatting()
        filtered_icustay = self._filter_icustay(df=self.resources.icustay)
        filtered_bg = self._filter_bg(
            bg=self.resources.bg, icustay=filtered_icustay
        )
        filtered_lab = self._filter_lab(
            lab=self.resources.lab, icustay=filtered_icustay
        )
        filtered_vital = self._filter_vital(
            vital=self.resources.vital, icustay=filtered_icustay
        )

        return pre.NewPrefilterOutput(
            icustay=filtered_icustay,
            bg=filtered_bg,
            vital=filtered_vital,
            lab=filtered_lab
        )


if __name__ == "__main__":
    resource_refs = pic.PrefilterResourceRefs()

    prefilter_resources = pre.NewPrefilterResources(
        icustay=pd.read_csv(resource_refs.icustay),
        bg=pd.read_csv(resource_refs.bg),
        vital=pd.read_csv(resource_refs.vital),
        lab=pd.read_csv(resource_refs.lab),
    )

    prefilter = NewPrefilter(resources=prefilter_resources)

    result = prefilter.process()
