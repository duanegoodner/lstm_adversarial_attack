import sys
from pathlib import Path
import project_config as pc
from query_db.mimiciii_database import MimiciiiDatabaseAccess

# sys.path.append(str(Path(__file__).parent))
# sys.path.append(str(Path(__file__).parent / "preprocess"))

if __name__ == "__main__":
    db_access = MimiciiiDatabaseAccess(
        dotenv_path=pc.DB_DOTENV_PATH, output_dir=pc.DB_OUTPUT_DIR
    )
    db_access.connect()
    db_query_results = db_access.run_sql_queries(
        sql_query_paths=pc.DB_QUERIES
    )
    db_access.close_connection()

    # preprocessor = ImplementedPreprocessor()
    # preprocessed_files = preprocessor.preprocess()








