import optuna
from optuna.storages import RDBStorage
from urllib.parse import quote
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.resource_io as rio
from pathlib import Path

with Path("/run/secrets/tuner_password").open(mode="r") as in_file:
    tuner_password = in_file.read()
encoded_password = quote(tuner_password, safe="")


db_url = f"postgresql://tuner:{encoded_password}@postgres_optuna/model_tuning"
storage = RDBStorage(url=db_url)


# study_id = storage.create_new_study(
#     directions=[optuna.study.StudyDirection.MINIMIZE],
#     study_name="model_tuning_01",
# )
#
# study_from_pickle = rio.ResourceImporter().import_pickle_to_object(
#     path=cfp.HYPERPARAMETER_OUTPUT_DIR
#     / "continued_trials"
#     / "checkpoints_tuner"
#     / "optuna_study.pickle"
# )
#
# for trial in study_from_pickle.get_trials():
#     storage.create_new_trial(study_id=study_id, template_trial=trial)

study_summaries = optuna.study.get_all_study_summaries(storage=storage)
study_names = [item.study_name for item in study_summaries]
full_studies = [
    optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True
    )
    for study_name in study_names
]
