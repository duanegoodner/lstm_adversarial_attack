from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HYPERPARAMETER_TUNING_OUT_DIR = DATA_DIR / "hyperparameter_tuning"


# DATABASE
DB_DOTENV_PATH = PROJECT_ROOT / "config" / "mimiciii_database.env"
DB_DEFAULT_QUERY_DIR = PROJECT_ROOT / "src" / "mimiciii_queries"
DB_QUERY_PATHS = [
    DB_DEFAULT_QUERY_DIR / "icustay_detail.sql",
    # DB_DEFAULT_QUERY_DIR / "pivoted_bg.sql",
    # DB_DEFAULT_QUERY_DIR / "pivoted_lab.sql",
    # DB_DEFAULT_QUERY_DIR / "pivoted_vital.sql",
]
DB_OUTPUT = DATA_DIR / "mimiciii_query_results"


# PREPROCESS
PREPROCESS_CHECKPOINTS = DATA_DIR / "preprocess_checkpoints"

PREFILTER_OUTPUT = PREPROCESS_CHECKPOINTS / "1_prefilter"
PREFILTER_OUTPUT_FILES = {
    "icustay": "icustay.pickle",
    "bg": "bg.pickle",
    "lab": "lab.pickle",
    "vital": "vital.pickle"
}

STAY_MEASUREMENT_OUTPUT = PREPROCESS_CHECKPOINTS / "2_merged_stay_measurements"
STAY_MEASUREMENT_OUTPUT_FILES = {
    "icustay_bg_lab_vital": "icustay_bg_lab_vital.pickle",
    "bg_lab_vital_summary_stats": "bg_lab_vital_summary_stats.pickle"
}

FULL_ADMISSION_LIST_OUTPUT = PREPROCESS_CHECKPOINTS / "3_full_admission_list"
FULL_ADMISSION_LIST_OUTPUT_FILES = {
    "full_admission_list": "full_admission_list.pickle"
}

FEATURE_BUILDER_OUTPUT = PREPROCESS_CHECKPOINTS / "4_feature_builder"
FEATURE_BUILDER_OUTPUT_FILES = {
    "hadm_list_with_processed_dfs": "hadm_list_with_processed_dfs.pickle"
}
DEFAULT_WINSORIZE_LOW="5%"
DEFAULT_WINSORIZE_HIGH="95%"
DEFAULT_RESAMPLE_INTERPOLATION="linear"
DEFAULT_RESAMPLE_LIMIT_DIRECTION="both"

PREPROCESS_OUTPUT_DIR = DATA_DIR / "output_feature_finalizer"
PREPROCESS_OUTPUT_FILES = {
    "measurement_data_list": "measurement_data_list.pickle",
    "in_hospital_mortality_list": "in_hospital_mortality_list.pickle",
}
MIN_OBSERVATION_HOURS = 1
MAX_OBSERVATION_HOURS = 48
REQUIRE_EXACT_NUM_HOURS = False
OBSERVATION_WINDOW_START = "intime"


