import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as lcs


@dataclass
class PrefilterResourceRefs:
    """
    Container for paths to files used by Prefilter
    """
    icustay: Path = cfg_paths.DB_OUTPUT_DIR / "icustay_detail.csv"
    bg: Path = cfg_paths.DB_OUTPUT_DIR / "pivoted_bg.csv"
    vital: Path = cfg_paths.DB_OUTPUT_DIR / "pivoted_vital.csv"
    lab: Path = cfg_paths.DB_OUTPUT_DIR / "pivoted_lab.csv"


@dataclass
class PrefilterSettings:
    """
    Container for objects imported by Prefilter
    """
    output_dir: Path = cfg_paths.PREFILTER_OUTPUT
    min_age: int = 18
    min_los_hospital: int = 1
    min_los_icu: int = 1
    bg_data_cols: list[str] = None
    lab_data_cols: list[str] = None
    vital_data_cols: list[str] = None

    def __post_init__(self):
        if self.bg_data_cols is None:
            self.bg_data_cols = lcs.PREPROCESS_BG_DATA_COLS
        if self.lab_data_cols is None:
            self.lab_data_cols = lcs.PREPROCESS_LAB_DATA_COLS
        if self.vital_data_cols is None:
            self.vital_data_cols = lcs.PREPROCESS_VITAL_DATA_COLS


@dataclass
class ICUStayMeasurementCombinerResourceRefs:
    """
    Container for paths to files imported by ICUStayMeasurementCombiner
    """
    icustay: Path = (
        cfg_paths.PREFILTER_OUTPUT
        / cfg_paths.PREFILTER_OUTPUT_FILES["icustay"]
    )
    bg: Path = (
        cfg_paths.PREFILTER_OUTPUT / cfg_paths.PREFILTER_OUTPUT_FILES["bg"]
    )
    lab: Path = (
        cfg_paths.PREFILTER_OUTPUT / cfg_paths.PREFILTER_OUTPUT_FILES["lab"]
    )
    vital: Path = (
        cfg_paths.PREFILTER_OUTPUT / cfg_paths.PREFILTER_OUTPUT_FILES["vital"]
    )


@dataclass
class ICUStayMeasurementCombinerSettings:
    """
    Container for ICUStayMeasurementCombiner config settings
    """
    output_dir: Path = cfg_paths.STAY_MEASUREMENT_OUTPUT
    # winsorize_upper: float = 0.95
    # winsorize_lower: float = 0.05
    bg_data_cols: list[str] = None
    lab_data_cols: list[str] = None
    vital_data_cols: list[str] = None

    def __post_init__(self):
        if self.bg_data_cols is None:
            self.bg_data_cols = lcs.PREPROCESS_BG_DATA_COLS
        if self.lab_data_cols is None:
            self.lab_data_cols = lcs.PREPROCESS_LAB_DATA_COLS
        if self.vital_data_cols is None:
            self.vital_data_cols = lcs.PREPROCESS_VITAL_DATA_COLS

    @property
    def all_measurement_cols(self) -> list[str]:
        return self.bg_data_cols + self.lab_data_cols + self.vital_data_cols


@dataclass
class FullAdmissionListBuilderResourceRefs:
    """
    Container for paths to files imported by FullAdmissionListBuilder
    """
    icustay_bg_lab_vital: Path = (
        cfg_paths.STAY_MEASUREMENT_OUTPUT
        / cfg_paths.STAY_MEASUREMENT_OUTPUT_FILES["icustay_bg_lab_vital"]
    )


@dataclass
class FullAdmissionListBuilderSettings:
    """
    Container for FullAdmissionListBuilder config settings
    """
    output_dir: Path = cfg_paths.FULL_ADMISSION_LIST_OUTPUT
    measurement_cols: list[str] = None

    def __post_init__(self):
        if self.measurement_cols is None:
            self.measurement_cols = (
                lcs.PREPROCESS_BG_DATA_COLS
                + lcs.PREPROCESS_LAB_DATA_COLS
                + lcs.PREPROCESS_VITAL_DATA_COLS
            )

    @property
    def time_series_cols(self) -> list[str]:
        return ["charttime"] + self.measurement_cols


@dataclass
class FullAdmissionData:
    """
    Container used as elements list build by FullAdmissionListBuilder
    """
    subject_id: np.ndarray
    hadm_id: np.ndarray
    icustay_id: np.ndarray
    admittime: np.ndarray
    dischtime: np.ndarray
    hospital_expire_flag: np.ndarray
    intime: np.ndarray
    outtime: np.ndarray
    time_series: pd.DataFrame


#  https://stackoverflow.com/a/65392400  (need this to work with dill)
FullAdmissionData.__module__ = __name__


@dataclass
class FeatureBuilderResourceRefs:
    """
    Container for paths to files imported by FeatureBuilder
    """
    full_admission_list: Path = (
        cfg_paths.FULL_ADMISSION_LIST_OUTPUT
        / cfg_paths.FULL_ADMISSION_LIST_OUTPUT_FILES["full_admission_list"]
    )
    bg_lab_vital_summary_stats: Path = (
        cfg_paths.STAY_MEASUREMENT_OUTPUT
        / cfg_paths.STAY_MEASUREMENT_OUTPUT_FILES["bg_lab_vital_summary_stats"]
    )


@dataclass
class FeatureBuilderSettings:
    """
    Container for FeatureBuilder config settings
    """
    output_dir: Path = cfg_paths.FEATURE_BUILDER_OUTPUT
    winsorize_low: str = lcs.DEFAULT_WINSORIZE_LOW
    winsorize_high: str = lcs.DEFAULT_WINSORIZE_HIGH
    resample_interpolation_method: str = lcs.DEFAULT_RESAMPLE_INTERPOLATION
    resample_limit_direction: str = lcs.DEFAULT_RESAMPLE_LIMIT_DIRECTION


@dataclass
class FeatureFinalizerResourceRefs:
    """
    Container for paths to files imported by FeatureFinalizer
    """
    processed_admission_list: Path = (
        cfg_paths.FEATURE_BUILDER_OUTPUT
        / cfg_paths.FEATURE_BUILDER_OUTPUT_FILES[
            "hadm_list_with_processed_dfs"
        ]
    )


@dataclass
class FeatureFinalizerSettings:
    """
    Container for FeatureFinalizer config settings
    """
    output_dir: Path = cfg_paths.PREPROCESS_OUTPUT_DIR
    max_hours: int = lcs.MAX_OBSERVATION_HOURS
    min_hours: int = lcs.MIN_OBSERVATION_HOURS
    require_exact_num_hours: bool = (
        lcs.REQUIRE_EXACT_NUM_HOURS  # when True, no need for padding
    )
    observation_window_start: str = lcs.OBSERVATION_WINDOW_START
