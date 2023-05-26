#!/bin/bash


trap 'true' SIGTERM
trap 'true' SIGINT

# TODO Move this to end of setup.sh
# psql -U postgres -d mimic -c "GRANT USAGE ON ALL TABLES IN SCHEMA mimiciii TO mimic;"
# psql -U postgres -d mimic -c "GRANT SELECT ON ALL TABLES IN SCHEMA mimiciii TO mimic;"


tail -f /dev/null &
wait $!

sudo chmod 775 -R $PGDATA
sudo chown $LOCAL_DB_OWNER:$LOCAL_DB_OWNER -R $PGDATA
