import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic
import lstm_adversarial_attack.preprocess.preprocess_module as ppm
import lstm_adversarial_attack.resource_io as rio


@dataclass
class PrefilterResources:
    """
    Container for the data frames Prefilter object imports and manipulates
    """

    icustay: pd.DataFrame
    bg: pd.DataFrame
    vital: pd.DataFrame
    lab: pd.DataFrame



class Prefilter(ppm.PreprocessModule):
    """
    Imports sql query output csvs to Dataframes and filters/formats dfs
    """

    def __init__(self):
        """
        Instantiates settings and resource references and passes to base class
        constructor
        """
        super().__init__(
            name="Prefilter",
            settings=pic.PrefilterSettings(),
            incoming_resource_refs=pic.PrefilterResourceRefs(),
        )

    @staticmethod
    def _apply_standard_df_formatting(
        imported_resources: PrefilterResources,
    ):
        """
        Sets col names of all dfs referenced by PrefilterResource to lowercase
        :param imported_resources: dataclass w/ refs to imported dataframes
        """
        for resource in imported_resources.__dict__.values():
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
            (df["admission_age"] >= self.settings.min_age)
            & (df["los_hospital"] >= self.settings.min_los_hospital)
            & (df["los_icu"] >= self.settings.min_los_icu)
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
            measurements_of_interest=self.settings.bg_data_cols,
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
            measurements_of_interest=self.settings.lab_data_cols,
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
            measurements_of_interest=self.settings.vital_data_cols,
        )

        return vital

    def _import_resources(self) -> PrefilterResources:
        """
        Imports csv data into dataframes and stores refs in a dataclass
        :return: a PrefilterResources dataclass containing dataframe refs
        """
        imported_data = PrefilterResources(
            icustay=pd.read_csv(self.incoming_resource_refs.icustay),
            bg=pd.read_csv(self.incoming_resource_refs.bg),
            vital=pd.read_csv(self.incoming_resource_refs.vital),
            lab=pd.read_csv(self.incoming_resource_refs.lab),
        )

        return imported_data

    def process(self):
        """
        Imports data, runs all filter methods, and exports filtered data.
        """
        imported_resources = self._import_resources()
        self._apply_standard_df_formatting(
            imported_resources=imported_resources
        )
        imported_resources.icustay = self._filter_icustay(
            df=imported_resources.icustay
        )
        imported_resources.bg = self._filter_bg(
            bg=imported_resources.bg,
            icustay=imported_resources.icustay,
        )
        imported_resources.lab = self._filter_lab(
            lab=imported_resources.lab,
            icustay=imported_resources.icustay,
        )
        imported_resources.vital = self._filter_vital(
            vital=imported_resources.vital,
            icustay=imported_resources.icustay,
        )

        for key, val in imported_resources.__dict__.items():
            self.export_resource_new(
                key=key,
                resource=val,
                path=self.settings.output_dir
                / cfg_paths.PREFILTER_OUTPUT_FILES[key],
                exporter=rio.df_to_json
            )


if __name__ == "__main__":
    prefilter = Prefilter()
    exported_resources = prefilter()
