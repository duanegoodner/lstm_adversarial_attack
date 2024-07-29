# ATTR_DISPLAY = {
#     "accuracy": "accuracy",
#     "auc": "AUC",
#     "f1": "F1",
#     "precision": "precision",
#     "recall": "recall",
#     "validation_loss": "_validation_loss",
# }

# PREPROCESS_BG_DATA_COLS = ["potassium", "calcium", "ph", "pco2", "lactate"]
# PREPROCESS_LAB_DATA_COLS = [
#     "albumin",
#     "bun",
#     "creatinine",
#     "sodium",
#     "bicarbonate",
#     "platelet",
#     "glucose",
#     "magnesium",
# ]
# PREPROCESS_VITAL_DATA_COLS = [
#     "heartrate",
#     "sysbp",
#     "diasbp",
#     "tempc",
#     "resprate",
#     "spo2",
# ]
# ADMISSION_DATA_JSON_DELIMITER = "END_TIME_SERIES_DF_METADATA"
# LSTM_INPUT_SIZE = (
#     len(PREPROCESS_BG_DATA_COLS)
#     + len(PREPROCESS_LAB_DATA_COLS)
#     + len(PREPROCESS_VITAL_DATA_COLS)
# )

# DEFAULT_WINSORIZE_LOW = "5%"
# DEFAULT_WINSORIZE_HIGH = "95%"
# DEFAULT_RESAMPLE_INTERPOLATION = "linear"
# DEFAULT_RESAMPLE_LIMIT_DIRECTION = "both"
#
# MIN_OBSERVATION_HOURS = 1
# observation_window_hours = 48
# REQUIRE_EXACT_NUM_HOURS = False
# OBSERVATION_WINDOW_START = "intime"

# TUNING_LOG_LSTM_HIDDEN_SIZE = (5, 7)
# TUNING_LSTM_ACT_OPTIONS = ("ReLU", "Tanh")
# TUNING_DROPOUT = (0, 0.5)
# TUNING_LOG_FC_HIDDEN_SIZE = (4, 8)
# TUNING_FC_ACT_OPTIONS = ("ReLU", "Tanh")
# TUNING_OPTIMIZER_OPTIONS = ("Adam", "RMSprop", "SGD")
# TUNING_LEARNING_RATE = (1e-5, 1e-1)
# TUNING_LOG_BATCH_SIZE = (5, 8)

# TUNER_NUM_FOLDS = 5
# TUNER_NUM_CV_EPOCHS = 20
# TUNER_EPOCHS_PER_FOLD = 5
# TUNER_NUM_FOLDS = 3
# TUNER_NUM_CV_EPOCHS = 2
# TUNER_EPOCHS_PER_FOLD = 2
# TUNER_PRUNER_NUM_STARTUP_TRIALS = 5
# TUNER_PRUNER_NUM_WARMUP_STEPS = 3
# TUNER_NUM_TRIALS = 30
# TUNER_KFOLD_RANDOM_SEED = 1234
# TUNER_CV_MEAN_TENSORBOARD_METRICS = (
#     "accuracy",
#     "auc",
#     "f1",
#     "precision",
#     "recall",
#     "validation_loss",
# )
# TUNER_PERFORMANCE_METRIC = "validation_loss"
# TUNER_OPTIMIZATION_DIRECTION = "minimize"
#
# TRAINER_RANDOM_SEED = 12345678
# TRAINER_EVAL_GENERAL_LOGGING_METRICS = (
#     "accuracy",
#     "auc",
#     "f1",
#     "precision",
#     "recall",
#     "validation_loss",
# )
# TRAINER_EVAL_TENSORBOARD_METRICS = (
#     # "accuracy",
#     "auc",
#     "f1",
#     "precision",
#     "recall",
#     "validation_loss",
# )

# CV_ASSESSMENT_RANDOM_SEED = 86420
# CV_DRIVER_EPOCHS_PER_FOLD = 20
# CV_DRIVER_NUM_FOLDS = 5
# CV_DRIVER_EVAL_INTERVAL = 10
# CV_DRIVER_SINGLE_FOLD_EVAL_FRACTION = 0.2

# ATTACK_TUNING_DEFAULT_NUM_TRIALS = 75
# ATTACK_TUNING_SAMPLE_SELECTION_SEED = 13579
# ATTACK_TUNING_DEFAULT_OBJECTIVE = "sparse_small_max"
# ATTACK_TUNING_KAPPA = (0.0, 2.0)
# ATTACK_TUNING_LAMBDA_1 = (1e-7, 1)
# ATTACK_TUNING_OPTIMIZER_OPTIONS = ("Adam", "RMSprop", "SGD")
# ATTACK_TUNING_LEARNING_RATE = (1e-5, 1)
# ATTACK_TUNING_LOG_BATCH_SIZE = (5, 7)
# ATTACK_TUNING_EPOCHS = 1000
# ATTACK_TUNING_MAX_NUM_SAMPLES = 520

# ATTACK_SAMPLE_SELECTION_SEED = 2023
# ATTACK_CHECKPOINT_INTERVAL = 50

# ATTACK_ANALYSIS_DEFAULT_SEQ_LENGTH = 48
