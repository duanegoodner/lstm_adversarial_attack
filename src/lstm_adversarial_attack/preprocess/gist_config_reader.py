import json
from pathlib import Path

with Path("gist_config.json").open(mode="r") as config_file:
    module_a_config = json.load(config_file)["module_a"]

print(module_a_config)