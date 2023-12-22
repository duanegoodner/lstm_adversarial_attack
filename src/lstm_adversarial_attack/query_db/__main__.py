import sys
from pathlib import Path

from lstm_adversarial_attack import config_paths

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config as cfr
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.query_db.mimiciii_database as mdb


def main() -> list[Path]:
    """
    Connects to MIMIC-III database, runs the .sql queries listed in
    cfg_paths.DB_QUERIES, and saves results as .csv files.
    :return: paths to the query output files
    """

    config_reader = cfr.ConfigReader()

    db_access = mdb.MimiciiiDatabaseAccess(
        dotenv_path=config_reader.read_path(
            config_key="paths.db.mimiciii_db_dotenv"),
        output_dir=config_reader.read_path(config_key="paths.db.query_output_dir"),
    )
    db_access.connect()
    db_query_results = db_access.run_sql_queries(
        sql_query_paths=config_reader.read_path(config_key="paths.db.queries")
    )
    db_access.close_connection()

    return db_query_results


if __name__ == "__main__":
    main()
