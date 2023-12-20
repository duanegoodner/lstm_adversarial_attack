from pathlib import Path
import os
import toml
import pprint


if __name__ == "__main__":
    with Path("config.toml").open(mode="r") as toml_file:
        config = toml.load(toml_file)
    pprint.pprint(config)
    print(os.getenv("CONTAINER_PROJECT_ROOT"))
