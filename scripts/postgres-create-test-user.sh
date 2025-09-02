#!/bin/bash
set -euo pipefail

# Helper script to create the ogd_weather_test user for integration tests.
# The idea is that this user has a dedicated ogd_weather_test schema
# in the DB owned by the main user ogd_weather.

DB_NAME="ogd_weather"
DB_HOST="localhost"
DB_PORT="5432"
DB_USER="ogd_weather_test"
DB_SCHEMA="ogd_weather_test"
DB_PASSWORD="$(openssl rand -hex 16)"

psql -U postgres -h "$DB_HOST" -p "$DB_PORT" -d ogd_weather -v ON_ERROR_STOP=1 \
  -v db_user="$DB_USER" -v db_password="$DB_PASSWORD" -v db_schema="$DB_SCHEMA" <<'SQL'
DROP SCHEMA IF EXISTS :"db_user" CASCADE;
DROP OWNED BY :"db_user";
DROP ROLE IF EXISTS :"db_user";
CREATE ROLE :"db_user" LOGIN PASSWORD :'db_password';
GRANT CONNECT ON DATABASE ogd_weather TO :"db_user";
CREATE SCHEMA IF NOT EXISTS :"db_schema" AUTHORIZATION :"db_user";
ALTER ROLE :"db_user" SET search_path = :"db_schema";
SQL

if [[ -f "$HOME/.pgpass" ]]; then
    echo "# $DB_HOST:$DB_PORT:$DB_NAME:$DB_USER:$DB_PASSWORD" >> "$HOME/.pgpass"
    echo "Added comment line to ~/.pgpass. Make sure to uncomment it and delete the old entry."
else
    echo "Created $DB_USER with password $DB_PASSWORD"
fi
