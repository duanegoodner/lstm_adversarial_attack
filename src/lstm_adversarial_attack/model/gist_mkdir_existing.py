from pathlib import Path
import lstm_adversarial_attack.config_paths as cfp


test_dir = cfp.DATA_DIR / "test_dir"
test_sub_dir = cfp.DATA_DIR / "test_dir" / "test_sub_dir"
test_sub_dir.mkdir(parents=True, exist_ok=True)
print(test_dir.exists())
print(test_sub_dir.exists())

test_dir.mkdir(parents=True, exist_ok=True)
print(test_sub_dir.exists())