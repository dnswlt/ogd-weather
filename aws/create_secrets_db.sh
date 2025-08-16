#!/usr/bin/env bash
set -euo pipefail

############################################################
### Postgres app user password
############################################################

SECRET_NAME="/weather/db/credentials/app_role"

# DB_USER should match the ROLE defined in ./bootstrap_db.sql
DB_USER="weather"

# Generate a strong password for the app user.
DB_PASSWORD="$(aws secretsmanager get-random-password \
  --password-length 32 \
  --require-each-included-type \
  --exclude-punctuation \
  --output text)"

# Use the same JSON format that AWS uses for the aws_db_instance.postgres.master_user_secret.
SECRET_STRING="{\"username\":\"$DB_USER\",\"password\":\"$DB_PASSWORD\"}"

# Store or update in Secrets Manager.
if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
  aws secretsmanager put-secret-value \
    --secret-id "$SECRET_NAME" \
    --secret-string "$SECRET_STRING" >/dev/null
else
  aws secretsmanager create-secret \
    --name "$SECRET_NAME" \
    --description "Postgres app role password" \
    --secret-string "$SECRET_STRING" >/dev/null
fi

echo "Updated Secrets Manager Postgres DB password: $SECRET_NAME"
