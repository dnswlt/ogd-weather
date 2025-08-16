#!/usr/bin/env bash
set -euo pipefail

### Store or update secret Postgres password in Secrets Manager.

SECRET_NAME="/weather/db/credentials"

# Generate a strong password for the app user.
DB_PASSWORD="$(aws secretsmanager get-random-password \
  --password-length 32 \
  --require-each-included-type \
  --exclude-punctuation \
  --output text)"

# Store or update in Secrets Manager.
if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
  aws secretsmanager put-secret-value \
    --secret-id "$SECRET_NAME" \
    --secret-string "$DB_PASSWORD" >/dev/null
else
  aws secretsmanager create-secret \
    --name "$SECRET_NAME" \
    --description "Postgres app user password" \
    --secret-string "$DB_PASSWORD" >/dev/null
fi

echo "Updated Secrets Manager Postgres DB password: $SECRET_NAME"
