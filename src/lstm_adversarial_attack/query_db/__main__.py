import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from lstm_adversarial_attack.config_paths import (
    DB_DOTENV_PATH,
    DB_OUTPUT_DIR,
)
from lstm_adversarial_attack.config_settings import DB_QUERIES
from lstm_adversarial_attack.query_db.mimiciii_database import (
    MimiciiiDatabaseAccess,
)


def main():
    db_access = MimiciiiDatabaseAccess(
        dotenv_path=DB_DOTENV_PATH, output_dir=DB_OUTPUT_DIR
    )
    db_access.connect()
    db_query_results = db_access.run_sql_queries(sql_query_paths=DB_QUERIES)
    db_access.close_connection()


if __name__ == "__main__":
    main()
