#!/bin/bash

sudo chown -R postgres:postgres /mimiciii_query_results

for file in /mimiciii_queries/*.sql; do
  echo Running query "$file"
  psql -U postgres -d mimic -f "$file"
done
