import pandas as pd
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.preprocess.preprocess_module as pm
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic
import lstm_adversarial_attack.resource_io as rio

@dataclass
class ICUStayMeasurementCombinerResources:
    """
    Container for the data ICUStayMeasurementCombiner object uses
    """
    icustay: pd.DataFrame
    bg: pd.DataFrame
    lab: pd.DataFrame
    vital: pd.DataFrame


class ICUStayMeasurementCombiner(pm.PreprocessModule):
    def __init__(
        self,
    ):
        """
        Instantiates settings and resource references and passes to base class
        constructor
        """
        super().__init__(
            name="ICU Stay Data + Measurement Data Combiner",
            settings=pic.ICUStayMeasurementCombinerSettings(),
            incoming_resource_refs=pic.ICUStayMeasurementCombinerResourceRefs(),
        )

    def _import_resources(self) -> ICUStayMeasurementCombinerResources:
        """
        Imports resources to dataframes and saves ref to each in a dataclass
        :return: ICUStayMeasurementCombinerResources (dataclass) w/ df refs
        """
        # imported_data = ICUStayMeasurementCombinerResources(
        #     icustay=self.import_pickle_to_df(
        #         self.incoming_resource_refs.icustay
        #     ),
        #     bg=self.import_pickle_to_df(self.incoming_resource_refs.bg),
        #     lab=self.import_pickle_to_df(self.incoming_resource_refs.lab),
        #     vital=self.import_pickle_to_df(self.incoming_resource_refs.vital),
        # )
        imported_data = ICUStayMeasurementCombinerResources(
            icustay=rio.json_to_df(path=self.incoming_resource_refs.bg),
            bg=rio.json_to_df(path=self.incoming_resource_refs.bg),
            lab=rio.json_to_df(path=self.incoming_resource_refs.lab),
            vital=rio.json_to_df(path=self.incoming_resource_refs.vital),
        )

        return imported_data

    def _create_id_bg(
        self, bg: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges bg dataframe with icustay dataframe
        :param bg: bg df (w/ filtering already applied by Prefilter)
        :param icustay: icustay df (w/ filtering already applied by Prefilter)
        :return: the merged (aka joined) dataframe
        """
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

    def _create_id_lab(
        self, lab: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges lab dataframe with icustay dataframe
        :param lab: lab df (w/ filtering already applied by Prefilter)
        :param icustay: icustay df (w/ filtering already applied by Prefilter)
        :return: the merged (aka joined) dataframe
        """
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
    def _create_id_vital(
        self, vital: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges vital dataframe with icustay dataframe
        :param vital: vital df (w/ filtering already applied by Prefilter)
        :param icustay: icustay df (w/ filtering already applied by Prefilter)
        :return: the merged (aka joined) dataframe
        """
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
    def _merge_measurement_sources(
        id_bg: pd.DataFrame,
        id_lab: pd.DataFrame,
        id_vital: pd.DataFrame,
    ):
        """
        Merges bg, lab, and vital data into single df
        :param id_bg: bg data previously merged with icustay
        :param id_lab: lab data previously merged with icustay
        :param id_vital: vital data previously merged with icustay
        :return: fully merged df
        """
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
    def _combine_icustay_info_with_measurements(
        id_bg_lab_vital: pd.DataFrame,
        icustay: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merges bg, lab and vital data with icustay data
        :param id_bg_lab_vital: df containing merged bg, lab, vital data
        :param icustay: icustay df
        :return: the merged df
        """
        info_id_bg_lab_vital = pd.merge(
            left=icustay,
            right=id_bg_lab_vital,
            on=["subject_id", "hadm_id", "icustay_id"],
            how="outer",
        )

        return info_id_bg_lab_vital

    def process(self):
        """
        Runs all private merge methods. exports fully merged df & summary df
        """
        data = self._import_resources()
        id_bg = self._create_id_bg(bg=data.bg, icustay=data.icustay)
        id_lab = self._create_id_lab(lab=data.lab, icustay=data.icustay)
        id_vital = self._create_id_vital(vital=data.vital, icustay=data.icustay)
        id_bg_lab_vital = self._merge_measurement_sources(
            id_bg=id_bg, id_lab=id_lab, id_vital=id_vital
        )
        icustay_bg_lab_vital = self._combine_icustay_info_with_measurements(
            id_bg_lab_vital=id_bg_lab_vital, icustay=data.icustay
        )

        summary_stats = icustay_bg_lab_vital[
            self.settings.all_measurement_cols
        ].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])

        self.export_resource(
            key="icustay_bg_lab_vital",
            resource=icustay_bg_lab_vital,
            path=self.settings.output_dir
            / cfg_paths.STAY_MEASUREMENT_OUTPUT_FILES["icustay_bg_lab_vital"],
        )

        self.export_resource(
            key="bg_lab_vital_summary_stats",
            resource=summary_stats,
            path=self.settings.output_dir
            / cfg_paths.STAY_MEASUREMENT_OUTPUT_FILES["bg_lab_vital_summary_stats"],
        )


if __name__ == "__main__":
    icustay_measurement_combiner = ICUStayMeasurementCombiner()
    exported_resources = icustay_measurement_combiner()
