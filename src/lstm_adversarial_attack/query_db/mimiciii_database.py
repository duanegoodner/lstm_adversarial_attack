import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from lstm_adversarial_attack.config.read_write import (
    CONFIG_READER,
    PATH_CONFIG_READER,
)


class MimiciiiDatabaseAccess:
    """
    Connects to and runs queries on MIMIC-III Postgres database
    """

    def __init__(self, dotenv_path: Path, output_parent: Path):
        """
        :param dotenv_path: path .env file w/ values needed for connection
        :param output_parent: parent of directory that will contain
        query output files
        """
        load_dotenv(dotenv_path=dotenv_path)
        self._connection = None
        self._dotenv_path = dotenv_path

        self._query_session_id = "".join(
            char for char in str(datetime.now()) if char.isdigit()
        )
        self._output_dir = output_parent / self._query_session_id
        self._output_dir.mkdir(parents=True)

    def connect(self):
        """
        Connects to database. connection object stored in self._connection.
        """
        load_dotenv(dotenv_path=self._dotenv_path)
        self._connection = psycopg2.connect(
            host=os.getenv("MIMICIII_DATABASE_HOST"),
            database=os.getenv("MIMICIII_DATABASE_NAME"),
            user=os.getenv("MIMICIII_DATABASE_USER"),
            password=os.getenv("MIMICIII_DATABASE_PASSWORD"),
        )

    def _execute_query(
        self, sql_file_path: Path
    ) -> tuple[list[str], list[tuple]]:
        """
        Executes sql query.
        :param sql_file_path:
        :type sql_file_path:
        :return: tuple (el[0] =  list of headers, el[1] = list of row results)
        """
        cur = self._connection.cursor()
        with sql_file_path.open(mode="r") as q:
            query = q.read()
        print(f"Executing: {sql_file_path}")
        start = time.time()
        cur.execute(query=query)
        result = cur.fetchall()
        headers = [i[0] for i in cur.description]
        end = time.time()
        print(f"Done. Query time = {(end - start):.2f} seconds")
        cur.close()
        return headers, result

    def _write_query_to_csv(
        self, query_result: tuple[list, list], query_gen_name: str
    ):
        """
        Saves result of query to .csv file.
        :param query_result: tuple of form output by self._execute_query()
        :param query_gen_name: label used for filename
        """
        output_path = self._output_dir / f"{query_gen_name}.csv"
        print(f"Writing result to csv: {output_path}")
        start = time.time()
        with output_path.open(mode="w", newline="") as q_out_file:
            writer = csv.writer(q_out_file, delimiter=",")
            writer.writerow(query_result[0])
            writer.writerows(query_result[1])
        end = time.time()
        print(f"Done. csv write time = {(end - start):.2f} seconds\n")
        # If csv writing is too slow, try modifying query text.
        # Would prepend with: "copy ("
        # And append with: "to str(output_path) with delimiter ',' csv header"

        # give file creator and work group full access to csv
        # (for work in app_dev container when connecting through Pycharm)
        # output_path.chmod(0o775)

    def _run_query_and_save_to_csv(self, sql_file_path: Path):
        """
        Runs a sql query and saves result to .csv.
        :param sql_file_path: path to .sql file
        """
        assert sql_file_path.name.endswith(".sql")
        query_gen_name = sql_file_path.name[: -len(".sql")]

        query_result = self._execute_query(sql_file_path=sql_file_path)
        self._write_query_to_csv(
            query_result=query_result, query_gen_name=query_gen_name
        )

    def run_sql_queries(self, sql_query_paths: list[Path]) -> list[Path]:
        """
        Runs and saves results of queries in a list of .sql filepaths.
        :param sql_query_paths: list of .sql query filepaths
        :return: list of paths to .csv query output files
        """

        print(f"Starting query session: {self._query_session_id}\n")

        CONFIG_READER.record_full_config(root_dir=self._output_dir)
        PATH_CONFIG_READER.record_full_config(root_dir=self._output_dir)

        result_paths = []
        for query_idx in range(len(sql_query_paths)):
            # for query_path in sql_query_paths:
            print(f"Query {query_idx + 1} of {len(sql_query_paths)}")
            query_path = sql_query_paths[query_idx]
            self._run_query_and_save_to_csv(sql_file_path=query_path)
            assert query_path.exists()
            result_paths.append(query_path)
        return result_paths

    def close_connection(self):
        """
        Closes database connection saved in self._connection
        """
        if self._connection is not None:
            self._connection.close()
