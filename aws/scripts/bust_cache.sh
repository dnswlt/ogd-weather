#!/usr/bin/env bash
set -euo pipefail

BEARER_TOKEN=$(aws ssm get-parameter --name /weather/api/BearerToken --with-decryption --output json | jq -r '.Parameter.Value')

URL_ENC="aHR0cHM6Ly93ZWF0aGVyLmhleHoubWUvYWRtaW4vY2FjaGUvcmVzcG9uc2VzCg=="
URL=$(echo "$URL_ENC" | openssl base64 -d -A)

echo "Busting cache at $URL"
curl -X DELETE -H "Authorization: Bearer $BEARER_TOKEN" "$URL"
