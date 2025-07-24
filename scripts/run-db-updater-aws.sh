#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <cluster-name> <task-family-name>" >&2
  exit 1
fi

CLUSTER="$1"
TASK_FAMILY="$2"

# Fetch subnets dynamically (or you can hardcode them if stable)
DEFAULT_SUBNETS=$(aws ec2 describe-subnets \
  --filters "Name=default-for-az,Values=true" \
  --region eu-central-1 --profile weather \
  --query "Subnets[].SubnetId" --output text)

if [ -z "$DEFAULT_SUBNETS" ]; then
  echo "No subnets found for ECS tasks" >&2
  exit 1
fi

# Turn into comma-separated list
SUBNETS_COMMA=$(echo "$DEFAULT_SUBNETS" | tr '\t' ',')

echo "Running latest revision of $TASK_FAMILY in cluster $CLUSTER ..."
aws ecs run-task \
  --cluster "$CLUSTER" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS_COMMA],assignPublicIp=ENABLED}" \
  --task-definition "$TASK_FAMILY"
