import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv


class DataExtractor:
    def __init__(
        self,
        # project_root: Path = Path(__file__).parent.parent.parent,
        # dot_env_rel_root: Path = Path("data/docker_db/.env"),
        queries_dir: Path = Path("/mimiciii_queries"),
        output_dir: Path = Path("/mimiciii_query_results"),
    ):
        # self.project_root = project_root
        # self.dot_env = project_root / dot_env_rel_root
        self.queries_dir = queries_dir
        self.output_dir = output_dir

    @property
    def all_query_files(self):
        return sorted(self.queries_dir.glob("*.sql"))

    def run_query(self, sql_file: Path):
        csv_filename = sql_file.name.replace(".sql", ".csv")
        csv_path = self.output_dir / csv_filename

        subprocess.run(
            [
                "psql",
                "-U",
                "postgres",
                "-d",
                "mimiciii",
                "-c",
                (
                    f"COPY ({open(sql_file).read()}) TO '{csv_path}' DELIMITER"
                    " ',' CSV HEADER"
                ),
            ],
            env={"PGPASSWORD": "postgres"},
            check=True,
        )


if __name__ == "__main__":
    data_extractor = DataExtractor()
    data_extractor.run_query(data_extractor.all_query_files[0])
