from src.ehr_adversarial_attack.resource_io import ResourceImporter

from preprocess.preprocess import \
    preprocess_input_classes as pic

features_path = pic.PREPROCESS_OUTPUT_DIR / "measurement_data.pickle"
in_hosp_mortality_path = (
    pic.PREPROCESS_OUTPUT_DIR / "in_hospital_mortality.pickle"
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

