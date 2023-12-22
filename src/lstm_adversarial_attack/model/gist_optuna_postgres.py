import optuna
from optuna.storages import RDBStorage


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


def main():
    # Replace these with your actual PostgreSQL database credentials and connection information
    db_url = "postgresql://optuna:optuna@postgres_optuna/optuna"

    storage = RDBStorage(url=db_url)
    # study = optuna.create_study(
    #     storage=storage, study_name="example_study", direction="minimize"
    # )
    # study.optimize(objective, n_trials=100)

    study = optuna.create_study(
        study_name="example_study", storage=storage, load_if_exists=True
    )
    study.optimize(objective, n_trials=5)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
