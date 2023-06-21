"""
Development script for examining .pickles saves by Preprocessor
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


features_path = cfg_paths.PREPROCESS_OUTPUT_DIR / "measurement_data_list.pickle"
in_hosp_mortality_path = (
    cfg_paths.PREPROCESS_OUTPUT_DIR / "in_hospital_mortality_list.pickle"
)
col_name_path = (
    cfg_paths.PREPROCESS_OUTPUT_DIR / "measurement_col_names.pickle"
)
importer = rio.ResourceImporter()
features = importer.import_pickle_to_object(path=features_path)
in_hosp_mortality = importer.import_pickle_to_object(
    path=in_hosp_mortality_path
)
col_names = importer.import_pickle_to_object(path=col_name_path)

