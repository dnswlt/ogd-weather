#!/usr/bin/env bash
set -euo pipefail

############################################################
### BearerToken for /status page access.
############################################################

if [[ "$#" -eq 0 ]]; then
    read -sp "Enter BearerToken: " PARAM_VALUE
else
    PARAM_VALUE="$1"
fi

PARAM_NAME="/weather/api/BearerToken"

aws ssm put-parameter \
  --name "$PARAM_NAME" \
  --value "$PARAM_VALUE" \
  --type SecureString \
  --tier Standard \
  --overwrite

echo "Updated SSM Parameter $PARAM_NAME"
