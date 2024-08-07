import time
from dataclasses import dataclass

import pandas as pd

import lstm_adversarial_attack.preprocess.preprocessor as pre
import lstm_adversarial_attack.preprocess.resource_data_structs as rds


@dataclass
class PrefilterSettings(pre.PreprocessModuleSettings):
    """
    Container for objects imported by Prefilter
    """

    min_age: int = None
    min_los_hospital: int = None
    min_los_icu: int = None
    bg_data_cols: list[str] = None
    lab_data_cols: list[str] = None
    vital_data_cols: list[str] = None


class Prefilter(pre.PreprocessModule):
    def __init__(
        self,
        resources: rds.PrefilterResources = None,
        settings: PrefilterSettings = None,
        output_constructors: rds.PrefilterOutputConstructors = None,
    ):
        if output_constructors is None:
            output_constructors = rds.PrefilterOutputConstructors()
        super().__init__(
            resources=resources,
            settings=settings,
            output_constructors=output_constructors,
        )
        self.icustay = resources.icustay.item
        self.bg = resources.bg.item
        self.vital = resources.vital.item
        self.lab = resources.lab.item

    def _apply_standard_formatting(self):
        """
        Sets col names of all dfs in self._resources to lowercase
        """

        for val in self.__dict__.values():
            if isinstance(val, pd.DataFrame):
                val.columns = [item.lower() for item in val.columns]

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

    def _filter_bg(self, bg: pd.DataFrame, icustay: pd.DataFrame) -> pd.DataFrame:
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

    def _filter_lab(self, lab: pd.DataFrame, icustay: pd.DataFrame) -> pd.DataFrame:
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

    def _filter_vital(self, vital: pd.DataFrame, icustay: pd.DataFrame) -> pd.DataFrame:
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
    ) -> dict[str, rds.OutgoingPreprocessResource]:
        """
        Runs all filter methods.
        """
        self._apply_standard_formatting()
        filtered_icustay = self._filter_icustay(df=self.icustay)
        filtered_bg = self._filter_bg(bg=self.bg, icustay=filtered_icustay)
        filtered_lab = self._filter_lab(lab=self.lab, icustay=filtered_icustay)
        filtered_vital = self._filter_vital(vital=self.vital, icustay=filtered_icustay)

        return {
            "prefiltered_icustay": self.output_constructors.prefiltered_icustay(
                resource=filtered_icustay
            ),
            "prefiltered_bg": self.output_constructors.prefiltered_bg(
                resource=filtered_bg
            ),
            "prefiltered_lab": self.output_constructors.prefiltered_lab(
                resource=filtered_lab
            ),
            "prefiltered_vital": self.output_constructors.prefiltered_vital(
                resource=filtered_vital
            ),
        }


if __name__ == "__main__":
    init_prefilter_start = time.time()
    prefilter_resources = rds.PrefilterResources(
        module_name="prefilter",
        default_data_source_type=rds.DataSourceType.FILE
    )

    prefilter = Prefilter(
        resources=prefilter_resources,
        settings=PrefilterSettings(module_name="prefilter"),
    )
    init_prefilter_end = time.time()
    print(f"prefilter init time = {init_prefilter_end - init_prefilter_start}")

    process_start = time.time()
    result = prefilter.process()
    process_end = time.time()
    print(f"prefilter process time = {process_end - process_start}")

    export_start = time.time()
    for key, outgoing_resource in result.items():
        outgoing_resource.export(
            path=prefilter.output_dir / f"{key}{outgoing_resource.file_ext}"
        )
    export_end = time.time()
    print(f"prefilter result export time = {export_end - export_start}")
