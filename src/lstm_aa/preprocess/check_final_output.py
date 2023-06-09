import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lstm_aa.resource_io import ResourceImporter

import preprocess_input_classes as pic

features_path = pic.PREPROCESS_OUTPUT_DIR / "measurement_data_list.pickle"
in_hosp_mortality_path = (
    pic.PREPROCESS_OUTPUT_DIR / "in_hospital_mortality_list.pickle"
)
col_name_path = (
    pic.PREPROCESS_OUTPUT_DIR / "measurement_col_names.pickle"
)

importer = ResourceImporter()
features = importer.import_pickle_to_object(path=features_path)
in_hosp_mortality = importer.import_pickle_to_object(
    path=in_hosp_mortality_path
)
col_names = importer.import_pickle_to_object(path=col_name_path)

