import sys
from pathlib  import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as lcp


ongoing_study_path = lcp.ONGOING_TUNING_STUDY_PICKLE

ongoing_study = rio.ResourceImporter().import_pickle_to_object(
    path=ongoing_study_path
)




