import pandas as pd
from dataclasses import dataclass
import preprocess_module as pm
import preprocess_input_classes as pic


@dataclass
class FeatureBuilderResources:
    # TODO type hint should be list[FullAdmissionData] (?)
    full_admission_list: pd.DataFrame


class FeatureBuilder(pm.PreprocessModule):
    def __init__(
        self,
        settings=pic.FeatureBuilderSettings(),
        incoming_resource_refs=pic.FeatureBuilderResourceRefs(),
    ):
        super().__init__(
            settings=settings, incoming_resource_refs=incoming_resource_refs
        )
        # since stats_summary df is small, make it a data member
        # (we try to keep scope of big dfs limited to process() method)
        stats_summary_path = incoming_resource_refs.bg_lab_vital_summary_stats
        self.stats_summary = self.import_pickle_to_df(stats_summary_path)

    def _import_resources(self) -> FeatureBuilderResources:
        imported_data = FeatureBuilderResources(
            full_admission_list=self.import_pickle_to_df(
                self.incoming_resource_refs.full_admission_list
            ),
        )
        return imported_data

    def _resample(
        self, raw_time_series_df: pd.DataFrame, raw_timestamp_col: str
    ) -> pd.DataFrame:
        resampled_df = (
            raw_time_series_df.resample("H")
            .mean()
            .interpolate(
                method=self.settings.resample_interpolation_method,
                limit_direction=self.settings.resample_limit_direction,
            )
        )
        return resampled_df

    def _fill_missing_data(self, resampled_df: pd.DataFrame):
        cols_with_all_nan = resampled_df.columns[resampled_df.isna().all()]
        na_fill_val_map = {
            col: self.stats_summary.loc["50%", col]
            for col in cols_with_all_nan
        }
        resampled_df.fillna(na_fill_val_map, inplace=True)

    def _winsorize(self, filled_df: pd.DataFrame) -> pd.DataFrame:
        winsorized_df = filled_df[self.stats_summary.columns].clip(
            lower=self.stats_summary.loc[self.settings.winsorize_low, :],
            upper=self.stats_summary.loc[self.settings.winsorize_high, :],
            axis=1,
        )
        return winsorized_df

    def _rescale(self, winsorized_df: pd.DataFrame) -> pd.DataFrame:
        rescaled_df = (
            winsorized_df
            - self.stats_summary.loc[self.settings.winsorize_low, :]
        ) / (
            self.stats_summary.loc[self.settings.winsorize_high, :]
            - self.stats_summary.loc[self.settings.winsorize_low, :]
        )
        return rescaled_df

    def _process_hadm_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # sorting probably not necessary, but helps with debugging
        df.sort_values("charttime")
        df.set_index("charttime", inplace=True)
        new_df = self._resample(
            raw_time_series_df=df, raw_timestamp_col="charttime"
        )
        self._fill_missing_data(resampled_df=new_df)
        new_df = self._winsorize(filled_df=new_df)
        new_df = self._rescale(winsorized_df=new_df)
        new_df.reset_index(inplace=True)

        return new_df

    def process(self):
        assert self.settings.output_dir.exists()

        data = self._import_resources()

        # filter out list elements with < 2 timestamps in their timeseries df
        data.full_admission_list = [
            entry
            for entry in data.full_admission_list
            if entry.time_series.shape[0] > 2
        ]

        for idx in range(len(data.full_admission_list)):
            data.full_admission_list[idx].time_series = self._process_hadm_df(
                df=data.full_admission_list[idx].time_series
            )
            if idx % 5000 == 0:
                print(
                    "Done processing sample"
                    f" {idx}/{len(data.full_admission_list)}"
                )

        # for hadm_entry in data.3_full_admission_list:
        #     hadm_entry.time_series = self._process_hadm_df(
        #         df=hadm_entry.time_series
        #     )

        self.export_resource(
            key="hadm_list_with_processed_dfs",
            resource=data.full_admission_list,
            path=self.settings.output_dir
            / "hadm_list_with_processed_dfs.pickle",
        )


if __name__ == "__main__":
    feature_builder = FeatureBuilder()
    exported_resources = feature_builder()
