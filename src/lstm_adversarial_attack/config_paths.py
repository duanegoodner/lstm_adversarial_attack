from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ##### Database #####
DB_DOTENV_PATH = PROJECT_ROOT / "config" / "mimiciii_database.env"
DB_DEFAULT_QUERY_DIR = PROJECT_ROOT / "src" / "mimiciii_queries"
DB_OUTPUT_DIR = DATA_DIR / "query_db"
DB_QUERIES = [
    DB_DEFAULT_QUERY_DIR / "icustay_detail.sql",
    DB_DEFAULT_QUERY_DIR / "pivoted_bg.sql",
    DB_DEFAULT_QUERY_DIR / "pivoted_lab.sql",
    DB_DEFAULT_QUERY_DIR / "pivoted_vital.sql",
]

# ##### Preprocessor #####
PREPROCESS_DATA_DIR = DATA_DIR / "preprocess"
PREPROCESS_CHECKPOINTS = PREPROCESS_DATA_DIR / "checkpoints"
PREFILTER_OUTPUT = PREPROCESS_CHECKPOINTS / "1_prefilter"
PREFILTER_OUTPUT_FILES = {
    "icustay": "icustay.pickle",
    "bg": "bg.pickle",
    "lab": "lab.pickle",
    "vital": "vital.pickle",
}
STAY_MEASUREMENT_OUTPUT = PREPROCESS_CHECKPOINTS / "2_merged_stay_measurements"
STAY_MEASUREMENT_OUTPUT_FILES = {
    "icustay_bg_lab_vital": "icustay_bg_lab_vital.pickle",
    "bg_lab_vital_summary_stats": "bg_lab_vital_summary_stats.pickle",
}
FULL_ADMISSION_LIST_OUTPUT = PREPROCESS_CHECKPOINTS / "3_full_admission_list"
FULL_ADMISSION_LIST_OUTPUT_FILES = {
    "full_admission_list": "full_admission_list.pickle"
}
FEATURE_BUILDER_OUTPUT = PREPROCESS_CHECKPOINTS / "4_feature_builder"
FEATURE_BUILDER_OUTPUT_FILES = {
    "hadm_list_with_processed_dfs": "hadm_list_with_processed_dfs.pickle"
}
PREPROCESS_OUTPUT_DIR = PREPROCESS_DATA_DIR / "final_output"
PREPROCESS_OUTPUT_FILES = {
    "measurement_data_list": "measurement_data_list.pickle",
    "in_hospital_mortality_list": "in_hospital_mortality_list.pickle",
}


TUNE_TRAIN_OUTPUT_DIR = DATA_DIR / "tune_train"


HYPERPARAMETER_OUTPUT_DIR = TUNE_TRAIN_OUTPUT_DIR / "hyperparameter_tuning"

# ##### Model Hyperparameter Tuning
ONGOING_TUNING_STUDY_DIR = HYPERPARAMETER_OUTPUT_DIR / "continued_trials"
ONGOING_TUNING_STUDY_PICKLE = (
    ONGOING_TUNING_STUDY_DIR / "checkpoints_tuner" / "optuna_study.pickle"
)
# CONTINUE_STUDY_PATH = (
#     HYPERPARAMETER_OUTPUT_DIR
#     / "continued_trials"
#     / "checkpoints_tuner"
#     / "optuna_study.pickle"
# )

# ##### Cross Validation Assessment Output #####
# MODEL_ASSESSMENTS_DIR = TUNE_TRAIN_OUTPUT_DIR / "model_assessments"
CV_ASSESSMENT_OUTPUT_DIR = TUNE_TRAIN_OUTPUT_DIR / "cross_validation"
# SINGLE_FOLD_OUTPUT_DIR = MODEL_ASSESSMENTS_DIR / "single_fold_training"


# DEFAULT_ATTACK_TARGET_DIR = SINGLE_FOLD_OUTPUT_DIR / "default_attack_target"

ATTACK_OUTPUT_DIR = DATA_DIR / "attack"
ATTACK_HYPERPARAMETER_TUNING = (
    ATTACK_OUTPUT_DIR / "attack_hyperparameter_tuning"
)
FROZEN_HYPERPARAMETER_ATTACK = (
    ATTACK_OUTPUT_DIR / "frozen_hyperparameter_attack"
)
