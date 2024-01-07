import pprint
from pathlib import Path

import lstm_adversarial_attack.config as config

if __name__ == "__main__":
    path_reader = config.PathReader(config_path=Path("gist_config_path.toml"))
    result = path_reader.read_path("more_paths.x")
    pprint.pprint(result)

    print()

    path_reader_b = config.PathReader(config_path=Path("paths.toml"))
    result_b = path_reader_b.read_path("preprocess.prefilter.output_dir")
    pprint.pprint(result_b)
