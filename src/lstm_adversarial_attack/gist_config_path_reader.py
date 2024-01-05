import pprint
from pathlib import Path

import lstm_adversarial_attack.config as config

if __name__ == "__main__":
    confg_reader = config.ConfigReader(config_path=Path("gist_config_path.toml"))
    result = confg_reader.read_path("more_paths.x")
    pprint.pprint(result)

    config_reader_b = config.ConfigReader(config_path=Path("config.toml"))
    result_b = config_reader_b.read_path("preprocess.output_dirs")
    pprint.pprint(result_b)
