import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from lstm_adversarial_attack.config.read_write import PATH_CONFIG_READER
import lstm_adversarial_attack.query_db.mimiciii_database as mdb

def test_mimiciii_db_connection():
    db_access = mdb.MimiciiiDatabaseAccess(
        dotenv_path=Path(PATH_CONFIG_READER.read_path(config_key="db.mimiciii_dotenv")),
        output_parent=Path(PATH_CONFIG_READER.read_path(config_key="db.output_root")),
    )

    try:
        db_access.connect()
        print("Successfully connected to MIMIC-III database.")
    except Exception as e:
        print(e)

    try:
        db_access.close_connection()
        print("Connection to MIMIC-III database successfully closed.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    test_mimiciii_db_connection()