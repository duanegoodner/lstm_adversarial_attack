import pandas as pd
import lstm_adversarial_attack.config_paths as cfp


bg_feather_import = pd.read_feather(
    path=cfp.PREFILTER_OUTPUT / "prefiltered_bg.feather"
)

bg_feather_import.set_index("index", inplace=True)

