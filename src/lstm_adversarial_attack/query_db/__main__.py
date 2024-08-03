import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config as cfr
import lstm_adversarial_attack.query_db.mimiciii_database as mdb
from lstm_adversarial_attack.config import CONFIG_READER


def main(query_path: str = None) -> list[Path]:
    """
    Connects to MIMIC-III database, runs the .sql queries listed in
    cfg_paths.DB_QUERIES, and saves results as .csv files.
    :return: paths to the query output files
    """

    config_reader = cfr.ConfigReader()

    if query_path is None:
        query_path = CONFIG_READER.read_path(config_key="db.query_files")

    db_access = mdb.MimiciiiDatabaseAccess(
        dotenv_path=Path(
            config_reader.read_path(config_key="db.mimiciii_dotenv")
        ),
        output_parent=Path(
            config_reader.read_path(config_key="db.output_root")
        ),
    )
    db_access.connect()
    db_query_results = db_access.run_sql_queries(
        sql_query_paths=[
            Path(item)
            for item in config_reader.read_path(config_key=query_path)
        ]
    )
    db_access.close_connection()

    return db_query_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs .sql queries on MIMIC-III database"
    )
    parser.add_argument(
        "-q",
        "--query_dir",
        type=str,
        action="store",
        nargs="?",
        help=f"Directory containing .sql query files. Defaults to path "
             f"specified by paths.db.output_root in config.toml"
    )
    args_namespace = parser.parse_args()
    main()


