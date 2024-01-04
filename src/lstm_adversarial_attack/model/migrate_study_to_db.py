# Code below no longer works after pickle removal, but keeping for details
# on how to migrate from pickle storage to database storage

# import optuna
# from optuna.storages import RDBStorage
#
# import lstm_adversarial_attack.config_paths as cfp
# import lstm_adversarial_attack.resource_io as rio

# study_from_pickle = rio.ResourceImporter().import_pickle_to_object(
#     path=cfp.HYPERPARAMETER_OUTPUT_DIR
#     / "continued_trials"
#     / "checkpoints_tuner"
#     / "optuna_study.pickle"
# )
#
# db_url = "postgresql://optuna:optuna@postgres_optuna/optuna"
# storage = RDBStorage(url=db_url)
# study_id = storage.create_new_study(
#     study_name="prediction_tuning_01",
#     directions=[optuna.study.StudyDirection.MINIMIZE],
# )
# for trial in study_from_pickle.get_trials():
#     storage.create_new_trial(study_id=study_id, template_trial=trial)
#
# study_from_db = optuna.load_study(
#     study_name="prediction_tuning_01", storage=storage
# )
