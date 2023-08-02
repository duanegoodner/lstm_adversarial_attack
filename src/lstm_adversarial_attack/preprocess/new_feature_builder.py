import pandas as pd
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfs
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic
import lstm_adversarial_attack.preprocess.new_preprocessor as pre


@dataclass
class NewFeatureBuilderSettings:
    """
    Container for FeatureBuilder config settings
    """

    winsorize_low: str = cfs.DEFAULT_WINSORIZE_LOW
    winsorize_high: str = cfs.DEFAULT_WINSORIZE_HIGH
    resample_interpolation_method: str = cfs.DEFAULT_RESAMPLE_INTERPOLATION
    resample_limit_direction: str = cfs.DEFAULT_RESAMPLE_LIMIT_DIRECTION


class NewFeatureBuilder(pre.AbstractFeatureBuilder):
    def __init__(
        self,
        resources: pre.NewFeatureBuilderResources,
        output_dir: Path = cfp.FEATURE_BUILDER_OUTPUT,
        settings: NewFeatureBuilderSettings = None,
    ):
        self.resources = resources
        self._output_dir = output_dir
        if settings is None:
            settings = NewFeatureBuilderSettings()
        self._settings = settings

    @property
    def settings(self) -> NewFeatureBuilderSettings:
        return self._settings

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def _resample(self, raw_time_series_df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples measurement times series to hourly increments.

        Imputes missing values by interpolation (for params w/ some data)
        :param raw_time_series_df: time series with actual measurement times
        :return: time series df with hourly values / averages
        """
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
        """
        Sets val of params with zero data for stay to global mean of that param
        :param resampled_df: df that may have no data in some mease cols
        """
        cols_with_all_nan = resampled_df.columns[resampled_df.isna().all()]
        na_fill_val_map = {
            col: self.resources.bg_lab_vital_summary_stats.loc["50%", col]
            for col in cols_with_all_nan
        }
        resampled_df.fillna(na_fill_val_map, inplace=True)

    def _winsorize(self, filled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorizes measurement data. Default is to 5th & 95th %ile
        :param filled_df: has all missing data imputed by interp or global mean
        """
        winsorized_df = filled_df[
            self.resources.bg_lab_vital_summary_stats.columns
        ].clip(
            lower=self.resources.bg_lab_vital_summary_stats.loc[
                self.settings.winsorize_low, :
            ],
            upper=self.resources.bg_lab_vital_summary_stats.loc[
                self.settings.winsorize_high, :
            ],
            axis=1,
        )
        return winsorized_df

    def _rescale(self, winsorized_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rescales all measurements so all data is between 0 and 1.
        :param winsorized_df: df with imputation and winsorization applied.
        """
        rescaled_df = (
            winsorized_df
            - self.resources.bg_lab_vital_summary_stats.loc[
                self.settings.winsorize_low, :
            ]
        ) / (
            self.resources.bg_lab_vital_summary_stats.loc[
                self.settings.winsorize_high, :
            ]
            - self.resources.bg_lab_vital_summary_stats.loc[
                self.settings.winsorize_low, :
            ]
        )
        return rescaled_df

    def _process_hadm_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes, winsorizes, and rescales data.
        :param df: dataframe member of FullAdmissionData object
        """
        # sorting probably not necessary, but helps with debugging
        df.sort_values("charttime")
        df.set_index("charttime", inplace=True)
        new_df = self._resample(
            raw_time_series_df=df,
            # raw_timestamp_col="charttime"
        )
        self._fill_missing_data(resampled_df=new_df)
        new_df = self._winsorize(filled_df=new_df)
        new_df = self._rescale(winsorized_df=new_df)
        new_df.reset_index(inplace=True)

        return new_df

    def process(self) -> pre.NewFeatureBuilderOutput:
        """
        Winsorizes, imputes and normalizes dfs in list of FullAdmission objects

        Exports result (full list of modified objects) to pickle)
        """

        # filter out list elements with < 2 timestamps in their timeseries df
        filtered_admission_list = [
            entry
            for entry in self.resources.admission_list
            if entry.time_series.shape[0] > 2
        ]

        for idx in range(len(filtered_admission_list)):
            filtered_admission_list[idx].time_series = self._process_hadm_df(
                df=filtered_admission_list[idx].time_series
            )
            if (idx + 1) % 5000 == 0:
                print(
                    "Done building features for sample"
                    f" {idx + 1}/{len(filtered_admission_list)}"
                )

        return pre.NewFeatureBuilderOutput(
            processed_admission_list=filtered_admission_list
        )
