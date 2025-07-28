#!/bin/bash
set -euo pipefail

# Helper script to create the ogd_weather on localhost.

# Configuration
DB_NAME="ogd_weather"
DB_USER="$DB_NAME"
DB_HOST="localhost"
DB_PORT="5432"
DB_PASSWORD=""

# Read password securely from stdin
read -s -p "Enter password for PostgreSQL new user $DB_USER: " DB_PASSWORD
echo

# Validate password
if [[ -z "$DB_PASSWORD" ]]; then
    echo "Password must not be empty"
    exit 1
fi
if [[ "$DB_PASSWORD" == *"'"* ]]; then
    echo "Password must not contain single quotes (')."
    exit 1
fi

# Create user and database
psql -U postgres -h "$DB_HOST" -p "$DB_PORT" -v ON_ERROR_STOP=1 <<EOF
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '$DB_USER') THEN
      CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
   END IF;
END
\$\$;
CREATE DATABASE $DB_NAME OWNER $DB_USER;
EOF

echo "Database '$DB_NAME' and user '$DB_USER' created successfully."
