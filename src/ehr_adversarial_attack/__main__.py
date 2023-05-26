from pathlib import Path
import project_config as pc
from mimiciii_database import MimiciiiDatabaseAccess
from preprocess.preprocess import ImplementedPreprocessor


if __name__ == "__main__":
    db_access = MimiciiiDatabaseAccess(
        dotenv_path=pc.DB_DOTENV_PATH, output_dir=pc.DB_OUTPUT
    )
    db_access.connect()
    db_query_results = db_access.run_sql_queries(
        sql_query_paths=pc.DB_QUERY_PATHS
    )
    db_access.close_connection()

    preprocessor = ImplementedPreprocessor()
    preprocessed_files = preprocessor.preprocess()








