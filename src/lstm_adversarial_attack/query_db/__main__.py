import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.query_db.mimiciii_database as mdb


def main() -> list[Path]:
    """
    Connects to MIMIC-III database, runs the .sql queries listed in
    cfg_paths.DB_QUERIES, and saves results as .csv files.
    :return: paths to the query output files
    """
    db_access = mdb.MimiciiiDatabaseAccess(
        dotenv_path=cfg_paths.DB_DOTENV_PATH,
        output_dir=cfg_paths.DB_OUTPUT_DIR,
    )
    db_access.connect()
    db_query_results = db_access.run_sql_queries(
        sql_query_paths=cfg_paths.DB_QUERIES
    )
    db_access.close_connection()

    return db_query_results


if __name__ == "__main__":
    main()
