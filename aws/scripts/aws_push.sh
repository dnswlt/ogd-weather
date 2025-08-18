#!/usr/bin/env bash
set -euo pipefail

# Script to push docker images built from the current commit to AWS.
# Usage: ./aws/scripts/aws_push.sh [TAG]

# cd to repo root
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TAG="${1:-$(git rev-parse --short=12 HEAD)}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION}}"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

docker push "${REGISTRY}/weather-api:${TAG}"
docker push "${REGISTRY}/weather-charts:${TAG}"
docker push "${REGISTRY}/weather-db-updater:${TAG}"

# Update terraform.tfvars
TEMP_TFVARS=$(mktemp)
TFVARS=./aws/terraform/terraform.tfvars
grep -E -v '^weather_(api|charts|db_updater)_version[[:space:]]' "$TFVARS" > "$TEMP_TFVARS"

echo "weather_api_version = \"$TAG\"" >> "$TEMP_TFVARS"
echo "weather_charts_version = \"$TAG\"" >> "$TEMP_TFVARS"
echo "weather_db_updater_version = \"$TAG\"" >> "$TEMP_TFVARS"

mv "$TEMP_TFVARS" "$TFVARS"

