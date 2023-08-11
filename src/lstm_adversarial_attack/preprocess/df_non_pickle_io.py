import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.resource_io as rio

df = rio.ResourceImporter().import_pickle_to_object(
    path=cfp.PREFILTER_OUTPUT / "icustay.pickle"
)

# my_json = prefiltered_icustay.to_json(orient="split", dtype="columns")





