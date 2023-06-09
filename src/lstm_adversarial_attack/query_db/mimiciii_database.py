import csv
import os
import psycopg2
import sys
import time
from dotenv import load_dotenv
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class MimiciiiDatabaseAccess:
    def __init__(self, dotenv_path: Path, output_dir: Path):
        load_dotenv(dotenv_path=dotenv_path)
        self._connection = None
        self._dotenv_path = dotenv_path
        self._output_dir = output_dir


    def connect(self):
        load_dotenv(dotenv_path=self._dotenv_path)
        self._connection = psycopg2.connect(
            host=os.getenv("MIMICIII_DATABASE_HOST"),
            database=os.getenv("MIMICIII_DATABASE_NAME"),
            user=os.getenv("MIMICIII_DATABASE_USER"),
            password=os.getenv("MIMICIII_DATABASE_PASSWORD"),
        )

    def _execute_query(self, sql_file_path: Path) -> tuple[list, list]:
        cur = self._connection.cursor()
        with sql_file_path.open(mode="r") as q:
            query = q.read()
        print(f"Executing query: {sql_file_path.name}")
        start = time.time()
        cur.execute(query=query)
        result = cur.fetchall()
        headers = [i[0] for i in cur.description]
        end = time.time()
        print(f"Done. Query time = {(end - start):.2f} seconds\n")
        cur.close()
        return headers, result

    def _write_query_to_csv(
        self, query_result: tuple[list, list], query_gen_name: str
    ):
        output_path = self._output_dir / f"{query_gen_name}.csv"
        print(f"Writing result to csv: {output_path.name}")
        start = time.time()
        with output_path.open(mode="w", newline="") as q_out_file:
            writer = csv.writer(q_out_file, delimiter=",")
            writer.writerow(query_result[0])
            writer.writerows(query_result[1])
        end = time.time()
        print(f"Done. csv write time = {(end - start):.2f} seconds\n")

        # give file creator and trusted work group full access to csv
        output_path.chmod(0o775)


    # If csv writing is too slow, try modifying query text.
    # Would prepend with: "copy ("
    # And append with: "to str(output_path) with delimiter ',' csv header"
    def _run_query_and_save_to_csv(self, sql_file_path: Path):
        assert sql_file_path.name.endswith(".sql")
        query_gen_name = sql_file_path.name[: -len(".sql")]

        query_result = self._execute_query(sql_file_path=sql_file_path)
        self._write_query_to_csv(
            query_result=query_result, query_gen_name=query_gen_name
        )

    def run_sql_queries(
        self, sql_query_paths: list[Path]
    ) -> list[Path]:
        result_paths = []
        for query_path in sql_query_paths:
            self._run_query_and_save_to_csv(sql_file_path=query_path)
            assert query_path.exists()
            result_paths.append(query_path)
        return result_paths

    def close_connection(self):
        if self._connection is not None:
            self._connection.close()


