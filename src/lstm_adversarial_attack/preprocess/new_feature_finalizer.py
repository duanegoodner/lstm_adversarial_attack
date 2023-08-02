from dataclasses import dataclass
from pathlib import Path
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfs



@dataclass
class FeatureFinalizerSettings:
    """
    Container for FeatureFinalizer config settings
    """
    output_dir: Path = cfp.PREPROCESS_OUTPUT_DIR
    max_hours: int = cfs.MAX_OBSERVATION_HOURS
    min_hours: int = cfs.MIN_OBSERVATION_HOURS
    require_exact_num_hours: bool = (
        cfs.REQUIRE_EXACT_NUM_HOURS  # when True, no need for padding
    )
    observation_window_start: str = cfs.OBSERVATION_WINDOW_START
