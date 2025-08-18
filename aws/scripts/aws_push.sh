#!/usr/bin/env bash
set -euo pipefail

# Script to push docker images built from the current commit to AWS.
# Usage: ./aws/scripts/aws_push.sh [TAG]

# cd to repo root
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Build the image tag from current UTC time and git commit hash,
# unless a tag is specified on the command line.
TS="$(date -u +%Y%m%d_%H%M)"
SHA="$(git rev-parse --short=7 HEAD)"
SHA_TAG="v${TS}_${SHA}"
TAG="${1:-$SHA_TAG}"

echo "Pushing images with tag $TAG."

# Require account id; resolve region with safe fallback
if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
  echo "AWS_ACCOUNT_ID not set."
  exit 1
fi
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
if [[ -z "$REGION" ]]; then
  echo "Neither AWS_REGION nor AWS_DEFAULT_REGION is set."
  exit 1
fi

REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

docker push "${REGISTRY}/weather-api:${TAG}"
docker push "${REGISTRY}/weather-charts:${TAG}"
docker push "${REGISTRY}/weather-db-updater:${TAG}"

# Update image tags in SSM parameter store
echo "Updating SSM parameters."
aws ssm put-parameter --name /weather/images/api/tag        --type String --value "$TAG" --overwrite
aws ssm put-parameter --name /weather/images/charts/tag     --type String --value "$TAG" --overwrite
aws ssm put-parameter --name /weather/images/db_updater/tag --type String --value "$TAG" --overwrite

echo "Release successfully pushed."
