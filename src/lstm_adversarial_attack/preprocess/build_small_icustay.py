import pandas as pd
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.resource_io as rio

full_icustay = pd.read_csv(
    filepath_or_buffer=cfp.PREFILTER_INPUT_FILES["icustay"]
)

icustay_500 = full_icustay.iloc[:500, :]

output_path = cfp.DB_OUTPUT_DIR / "icustay_500.csv"

icustay_500.to_csv(output_path)

