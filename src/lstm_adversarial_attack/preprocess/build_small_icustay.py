"""
Dev script for building small version of icustay data
"""

from pathlib import Path

import pandas as pd

from lstm_adversarial_attack.config import CONFIG_READER

if __name__ == "__main__":
    full_icustay_path = Path(
        CONFIG_READER.read_path("preprocess.prefilter.resources.icustay")
    )
    full_icustay = pd.read_csv(full_icustay_path)

    icustay_4000 = full_icustay.iloc[:4000, :]

    db_output_dir = Path(
        CONFIG_READER.read_path(config_key="db.paths.query_output_dir")
    )

    output_path = db_output_dir / "icustay_4000.csv"

    icustay_4000.to_csv(output_path)
