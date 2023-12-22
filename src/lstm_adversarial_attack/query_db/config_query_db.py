from dotenv import load_dotenv
from pathlib import Path


class ConfigQueryDB:
    def __init__(self):
        pass

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent.parent

