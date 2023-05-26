import os
from pathlib import Path

from dotenv import load_dotenv

os.environ["PROJECT_ROOT_DIR"] = str(Path(__file__).parent.parent.parent)

proj_root = os.getenv("PROJECT_ROOT_DIR")

print(proj_root)



