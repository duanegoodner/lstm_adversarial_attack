import csv

import psycopg2
from pathlib import Path


if __name__ == "__main__":
    conn = psycopg2.connect(
        host="postgres_mimiciii",
        database="mimic",
        user="postgres",
        password="postgres",
    )

    cur = conn.cursor()

    admissions_query_path = Path(
        __file__).parent / "mimiciii_queries" / "admissions.sql"

    with admissions_query_path.open(mode="r") as aq_file:
        admissions_query = aq_file.read()

    cur.execute(admissions_query)

    rows = cur.fetchall()

    admissions_query_output_path = (
            Path(__file__).parent.parent
            / "data"
            / "mimiciii_query_results"
            / "admissions.csv"
    )

    cur.close()
    conn.close()

    with admissions_query_output_path.open(mode="w",
                                           newline="") as aq_out_file:
        writer = csv.writer(aq_out_file)
        writer.writerow(rows)

