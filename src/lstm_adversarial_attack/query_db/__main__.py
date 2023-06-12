import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.query_db.mimiciii_database as mdb


def main():
    db_access = mdb.MimiciiiDatabaseAccess(
        dotenv_path=lcp.DB_DOTENV_PATH, output_dir=lcp.DB_OUTPUT_DIR
    )
    db_access.connect()
    db_query_results = db_access.run_sql_queries(
        sql_query_paths=lcp.DB_QUERIES
    )
    db_access.close_connection()


if __name__ == "__main__":
    main()
