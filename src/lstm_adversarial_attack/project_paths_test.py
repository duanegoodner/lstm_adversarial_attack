import os
from dotenv import load_dotenv


load_dotenv("project_paths.env")
thing = os.getenv("DB_QUERIES_TO_RUN")
print(thing)
