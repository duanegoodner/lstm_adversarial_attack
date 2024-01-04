import optuna
from optuna.storages import RDBStorage
from urllib.parse import quote
from pathlib import Path

with Path("/run/secrets/tuner_password").open(mode="r") as in_file:
    tuner_password = in_file.read()
encoded_password = quote(tuner_password, safe="")


db_url = f"postgresql://tuner:{encoded_password}@postgres_optuna/model_tuning"
storage = RDBStorage(url=db_url)

study_summaries = optuna.study.get_all_study_summaries(storage=storage)
study_names = [item.study_name for item in study_summaries]
full_studies = [
    optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True
    )
    for study_name in study_names
]
