import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tdb
from lstm_adversarial_attack.tuning_db.tuning_studies_database import (
    OptunaDatabase,
    MODEL_TUNING_DB,
    ATTACK_TUNING_DB,
)

def test_tuning_study_db(db: OptunaDatabase):
    try:
        tuning_studies = db.get_all_studies()
        print(f"{db.db_name} database successfully queried.\n"
              f"Found {len(tuning_studies)} tuning studies.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    test_tuning_study_db(db=MODEL_TUNING_DB)
    test_tuning_study_db(db=ATTACK_TUNING_DB)
