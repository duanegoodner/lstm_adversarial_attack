import pprint
from pathlib import Path

import lstm_adversarial_attack.config as config

if __name__ == "__main__":
    confg_reader = config.ConfigReader(config_path=Path("gist_config_path.toml"))
    result = confg_reader.read_dotted_info("more_paths.x")
    pprint.pprint(result)
