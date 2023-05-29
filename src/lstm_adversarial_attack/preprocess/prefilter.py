import pandas as pd
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from lstm_adversarial_attack.config_paths import PREFILTER_OUTPUT_FILES
from preprocess_module import PreprocessModule
from preprocess_input_classes import PrefilterSettings, PrefilterResourceRefs


@dataclass
class PrefilterResources:
    icustay: pd.DataFrame
    bg: pd.DataFrame
    vital: pd.DataFrame
    lab: pd.DataFrame


class Prefilter(PreprocessModule):
    def __init__(
        self,
        settings=PrefilterSettings(),
        incoming_resource_refs=PrefilterResourceRefs(),
    ):
        super().__init__(
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
        )

    @staticmethod
    def _apply_standard_df_formatting(
        imported_resources: PrefilterResources,
    ):
        for resource in imported_resources.__dict__.values():
            if isinstance(resource, pd.DataFrame):
                resource.columns = [item.lower() for item in resource.columns]

    def _filter_icustay(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df = df[identifier_cols + measurements_of_interest]
        df = df.dropna(subset=measurements_of_interest, how="all")
        return df

    def _filter_bg(
        self, bg: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
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
        vital["charttime"] = pd.to_datetime(vital["charttime"])
        vital = vital[vital["icustay_id"].isin(icustay["icustay_id"])]

        vital = self._filter_measurement_df(
            df=vital,
            identifier_cols=["icustay_id", "charttime"],
            measurements_of_interest=self.settings.vital_data_cols,
        )

        return vital

    def _import_resources(self) -> PrefilterResources:
        imported_data = PrefilterResources(
            icustay=self.import_csv(path=self.incoming_resource_refs.icustay),
            bg=self.import_csv(path=self.incoming_resource_refs.bg),
            vital=self.import_csv(path=self.incoming_resource_refs.vital),
            lab=self.import_csv(path=self.incoming_resource_refs.lab),
        )

        return imported_data

    def process(self):
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
            self.export_resource(
                key=key,
                resource=val,
                path=self.settings.output_dir / PREFILTER_OUTPUT_FILES[key],
            )


if __name__ == "__main__":
    prefilter = Prefilter()
    exported_resources = prefilter()
