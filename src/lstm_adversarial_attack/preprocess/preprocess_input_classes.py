import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.config_settings as lcs


BG_DATA_COLS = ["potassium", "calcium", "ph", "pco2", "lactate"]
LAB_DATA_COLS = [
    "albumin",
    "bun",
    "creatinine",
    "sodium",
    "bicarbonate",
    "platelet",
    "glucose",
    "magnesium",
]
VITAL_DATA_COLS = [
    "heartrate",
    "sysbp",
    "diasbp",
    "tempc",
    "resprate",
    "spo2",
]


@dataclass
class PrefilterResourceRefs:
    icustay: Path = lcp.DB_OUTPUT_DIR / "icustay_detail.csv"
    bg: Path = lcp.DB_OUTPUT_DIR / "pivoted_bg.csv"
    vital: Path = lcp.DB_OUTPUT_DIR / "pivoted_vital.csv"
    lab: Path = lcp.DB_OUTPUT_DIR / "pivoted_lab.csv"


@dataclass
class PrefilterSettings:
    output_dir: Path = lcp.PREFILTER_OUTPUT
    min_age: int = 18
    min_los_hospital: int = 1
    min_los_icu: int = 1
    bg_data_cols: list[str] = None
    lab_data_cols: list[str] = None
    vital_data_cols: list[str] = None

    def __post_init__(self):
        if self.bg_data_cols is None:
            self.bg_data_cols = BG_DATA_COLS
        if self.lab_data_cols is None:
            self.lab_data_cols = LAB_DATA_COLS
        if self.vital_data_cols is None:
            self.vital_data_cols = VITAL_DATA_COLS


@dataclass
class ICUStayMeasurementCombinerResourceRefs:
    icustay: Path = lcp.PREFILTER_OUTPUT / lcp.PREFILTER_OUTPUT_FILES["icustay"]
    bg: Path = lcp.PREFILTER_OUTPUT / lcp.PREFILTER_OUTPUT_FILES["bg"]
    lab: Path = lcp.PREFILTER_OUTPUT / lcp.PREFILTER_OUTPUT_FILES["lab"]
    vital: Path = lcp.PREFILTER_OUTPUT / lcp.PREFILTER_OUTPUT_FILES["vital"]


@dataclass
class ICUStayMeasurementCombinerSettings:
    output_dir: Path = lcp.STAY_MEASUREMENT_OUTPUT
    winsorize_upper: float = 0.95
    winsorize_lower: float = 0.05
    bg_data_cols: list[str] = None
    lab_data_cols: list[str] = None
    vital_data_cols: list[str] = None

    def __post_init__(self):
        if self.bg_data_cols is None:
            self.bg_data_cols = BG_DATA_COLS
        if self.lab_data_cols is None:
            self.lab_data_cols = LAB_DATA_COLS
        if self.vital_data_cols is None:
            self.vital_data_cols = VITAL_DATA_COLS

    @property
    def all_measurement_cols(self) -> list[str]:
        return self.bg_data_cols + self.lab_data_cols + self.vital_data_cols


@dataclass
class FullAdmissionListBuilderResourceRefs:
    icustay_bg_lab_vital: Path = (
        lcp.STAY_MEASUREMENT_OUTPUT
        / lcp.STAY_MEASUREMENT_OUTPUT_FILES["icustay_bg_lab_vital"]
    )


@dataclass
class FullAdmissionListBuilderSettings:
    output_dir: Path = lcp.FULL_ADMISSION_LIST_OUTPUT
    measurement_cols: list[str] = None

    def __post_init__(self):
        if self.measurement_cols is None:
            self.measurement_cols = (
                BG_DATA_COLS + LAB_DATA_COLS + VITAL_DATA_COLS
            )

    @property
    def time_series_cols(self) -> list[str]:
        return ["charttime"] + self.measurement_cols


@dataclass
class FullAdmissionData:
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
    full_admission_list: Path = (
        lcp.FULL_ADMISSION_LIST_OUTPUT
        / lcp.FULL_ADMISSION_LIST_OUTPUT_FILES["full_admission_list"]
    )
    bg_lab_vital_summary_stats: Path = (
        lcp.STAY_MEASUREMENT_OUTPUT
        / lcp.STAY_MEASUREMENT_OUTPUT_FILES["bg_lab_vital_summary_stats"]
    )


@dataclass
class FeatureBuilderSettings:
    output_dir: Path = lcp.FEATURE_BUILDER_OUTPUT
    winsorize_low: str = lcs.DEFAULT_WINSORIZE_LOW
    winsorize_high: str = lcs.DEFAULT_WINSORIZE_HIGH
    resample_interpolation_method: str = lcs.DEFAULT_RESAMPLE_INTERPOLATION
    resample_limit_direction: str = lcs.DEFAULT_RESAMPLE_LIMIT_DIRECTION


#     TODO add data member for cutoff time after admit or before discharge


@dataclass
class FeatureFinalizerResourceRefs:
    processed_admission_list: Path = (
        lcp.FEATURE_BUILDER_OUTPUT
        / lcp.FEATURE_BUILDER_OUTPUT_FILES["hadm_list_with_processed_dfs"]
    )


@dataclass
class FeatureFinalizerSettings:
    output_dir: Path = lcp.PREPROCESS_OUTPUT_DIR
    max_hours: int = lcs.MAX_OBSERVATION_HOURS
    min_hours: int = lcs.MIN_OBSERVATION_HOURS
    require_exact_num_hours: bool = (
        lcs.REQUIRE_EXACT_NUM_HOURS  # when True, no need for padding
    )
    observation_window_start: str = lcs.OBSERVATION_WINDOW_START
