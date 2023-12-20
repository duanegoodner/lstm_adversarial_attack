import yaml
from pathlib import Path

with Path("test_yaml.yaml").open(mode="r") as f:
    prime_service = yaml.safe_load(f)

print(prime_service)


