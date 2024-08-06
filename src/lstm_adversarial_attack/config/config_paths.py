from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
LSTM_ADVERSARIAL_ATTACK_PACKAGE = SRC_DIR / "lstm_adversarial_attack"
DATA_DIR = PROJECT_ROOT / "data"
EXAMPLE_DATA_DIR = DATA_DIR / "example_data"
DOCKER_DIR = PROJECT_ROOT / "docker"


# ##### Database #####
# DB_SUBPACKAGE = LSTM_ADVERSARIAL_ATTACK_PACKAGE / "query_db"
# MIMICIII_DB_DOTENV_PATH = DB_SUBPACKAGE / "mimiciii_database.env"
# DB_DEFAULT_QUERY_DIR = DB_SUBPACKAGE / "mimiciii_queries"
# DB_OUTPUT_DIR = DATA_DIR / "query_db"
# DB_QUERIES = [
#     DB_DEFAULT_QUERY_DIR / "icustay_detail.sql",
#     DB_DEFAULT_QUERY_DIR / "pivoted_bg.sql",
#     DB_DEFAULT_QUERY_DIR / "pivoted_lab.sql",
#     DB_DEFAULT_QUERY_DIR / "pivoted_vital.sql",
# ]
# TUNING_STUDY_DB_SUBPACKAGE = LSTM_ADVERSARIAL_ATTACK_PACKAGE / "tuning_db"
TUNING_DBS_DOTENV_PATH = DOCKER_DIR / "tuning_dbs.env"


# ##### Preprocessor #####
# PREPROCESS_DATA_DIR = DATA_DIR / "preprocess"
# PREPROCESS_CHECKPOINTS = PREPROCESS_DATA_DIR / "checkpoints"
# PREFILTER_OUTPUT = PREPROCESS_CHECKPOINTS / "1_prefilter"
# PREFILTER_INPUT_FILES = {
#     "icustay": DB_OUTPUT_DIR / "icustay_detail.csv",
#     "bg": DB_OUTPUT_DIR / "pivoted_bg.csv",
#     "vital": DB_OUTPUT_DIR / "pivoted_vital.csv",
#     "lab": DB_OUTPUT_DIR / "pivoted_lab.csv",
# }
# STAY_MEASUREMENT_INPUT_FILES = {
#     "prefiltered_icustay": PREFILTER_OUTPUT / "prefiltered_icustay.feather",
#     "prefiltered_bg": PREFILTER_OUTPUT / "prefiltered_bg.feather",
#     "prefiltered_lab": PREFILTER_OUTPUT / "prefiltered_lab.feather",
#     "prefiltered_vital": PREFILTER_OUTPUT / "prefiltered_vital.feather",
# }
# STAY_MEASUREMENT_OUTPUT = PREPROCESS_CHECKPOINTS / "2_merged_stay_measurements"
# STAY_MEASUREMENT_OUTPUT_FILES = {
#     "icustay_bg_lab_vital": "icustay_bg_lab_vital.pickle",
#     "bg_lab_vital_summary_stats": "bg_lab_vital_summary_stats.pickle",
# }
#
# FULL_ADMISSION_LIST_INPUT_FILES = {
#     "icustay_bg_lab_vital": (
#         STAY_MEASUREMENT_OUTPUT / "icustay_bg_lab_vital.feather"
#     )
# }
# FULL_ADMISSION_LIST_OUTPUT = PREPROCESS_CHECKPOINTS / "3_full_admission_list"
# FULL_ADMISSION_LIST_OUTPUT_FILES = {
#     "full_admission_list": "full_admission_list.pickle"
# }
#
# FEATURE_BUILDER_OUTPUT = PREPROCESS_CHECKPOINTS / "4_feature_builder"
# FEATURE_BUILDER_INPUT_FILES = {
#     "bg_lab_vital_summary_stats": (
#         STAY_MEASUREMENT_OUTPUT / "bg_lab_vital_summary_stats.feather"
#     ),
#     "full_admission_list": (
#         FULL_ADMISSION_LIST_OUTPUT / "full_admission_list.json"
#     ),
# }
# FEATURE_BUILDER_OUTPUT_FILES = {
#     "hadm_list_with_processed_dfs": "hadm_list_with_processed_dfs.pickle"
# }
#
# FEATURE_FINALIZER_INPUT_FILES = {
#     "processed_admission_list": (
#         FEATURE_BUILDER_OUTPUT / "processed_admission_list.json"
#     )
# }
# FEATURE_FINALIZER_OUTPUT = PREPROCESS_CHECKPOINTS / "5_feature_finalizer"
#
# PREPROCESS_OUTPUT_DIR = FEATURE_FINALIZER_OUTPUT
# PREPROCESS_OUTPUT_FILES = {
#     "measurement_data_list": "measurement_data_list.json",
#     "in_hospital_mortality_list": "in_hospital_mortality_list.json",
#     "measurement_col_names": "measurement_col_names.json"
# }


TUNE_TRAIN_OUTPUT_DIR = DATA_DIR / "model"

# ##### Model Hyperparameter Tuning
HYPERPARAMETER_OUTPUT_DIR = TUNE_TRAIN_OUTPUT_DIR / "hyperparameter_tuning"
ONGOING_TUNING_STUDY_DIR = HYPERPARAMETER_OUTPUT_DIR / "continued_trials"
ONGOING_TUNING_STUDY_PICKLE = (
    ONGOING_TUNING_STUDY_DIR / "checkpoints_tuner" / "optuna_study.pickle"
)

# ##### Cross Validation Assessment Output #####
# CV_ASSESSMENT_OUTPUT_DIR = TUNE_TRAIN_OUTPUT_DIR / "cross_validation"


# DEFAULT_ATTACK_TARGET_DIR = SINGLE_FOLD_OUTPUT_DIR / "default_attack_target"
# ATTACK_DEFAULT_TARGET_MODEL_DIR = (
#     EXAMPLE_DATA_DIR / "cv_training_2023-06-17_23_57_23.366142"
# )
ATTACK_OUTPUT_DIR = DATA_DIR / "attack"
ATTACK_HYPERPARAMETER_TUNING = (
    ATTACK_OUTPUT_DIR / "tuning"
)
# FROZEN_HYPERPARAMETER_ATTACK = (
#     ATTACK_OUTPUT_DIR / "frozen_hyperparameter_attack"
# )

# ATTACK_ANALYSIS_DIR = DATA_DIR / "attack_analysis"
