"""
Dev script for building small version of icustay data
"""
from pathlib import Path

import pandas as pd

import lstm_adversarial_attack.config as config

if __name__ == "__main__":
    config_reader = config.ConfigReader()
    full_icustay_path = Path(
        config_reader.read_path("preprocess.prefilter.resources.icustay")
    )
    full_icustay = pd.read_csv(full_icustay_path)

    icustay_4000 = full_icustay.iloc[:4000, :]

    db_output_dir = Path(
        config_reader.read_path(config_key="db.paths.query_output_dir")
    )

    output_path = db_output_dir / "icustay_4000.csv"

    icustay_4000.to_csv(output_path)
