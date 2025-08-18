#!/usr/bin/env bash
set -euo pipefail

# Change to the repo root
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Builds and tags AWS docker images.
# The repo must be in a clean state, or else the script will abort.

# Region: use AWS_REGION, else AWS_DEFAULT_REGION
if [[ -n "${AWS_REGION:-}" ]]; then
    AWS_REGION="${AWS_REGION}"
elif [[ -n "${AWS_DEFAULT_REGION:-}" ]]; then
    AWS_REGION="${AWS_DEFAULT_REGION}"
else
    echo "Neither AWS_REGION nor AWS_DEFAULT_REGION are set."
    exit 1
fi

# Make the AWS account explicit (cleaner failure than unbound var)
if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
    echo "AWS_ACCOUNT_ID not set."
    exit 1
fi

# Abort if the git repo has local changes.
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Repository is not clean. Commit or stash changes before building."
    exit 1
fi

# Used for the Go api server as --build-arg flags.
VERSION="$(git describe --tags --always 2>/dev/null || echo dev)"
COMMIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BUILDTIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$COMMIT" = "unknown" ]]; then
    echo "Failed to get git commit hash for image tagging."
    exit 1
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="$COMMIT"

CHARTS_IMAGE="${ECR_REGISTRY}/weather-charts:${IMAGE_TAG}"
API_IMAGE="${ECR_REGISTRY}/weather-api:${IMAGE_TAG}"
DB_UPDATER_IMAGE="${ECR_REGISTRY}/weather-db-updater:${IMAGE_TAG}"

echo "Building docker images with tag ${IMAGE_TAG}"

docker build -t "${CHARTS_IMAGE}" -f service/charts/Dockerfile .
docker build \
    --build-arg "VERSION=${VERSION}" \
    --build-arg "COMMIT=${COMMIT}" \
    --build-arg "BUILDTIME=${BUILDTIME}" \
    -t "${API_IMAGE}" -f service/api/Dockerfile ./service/api
docker build -t "${DB_UPDATER_IMAGE}" -f service/db_updater/Dockerfile .
