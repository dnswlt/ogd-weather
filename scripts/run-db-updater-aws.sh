#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <cluster-name> <task-family-name> [extra aws ecs run-task args...]" >&2
  exit 1
fi

CLUSTER="$1"
TASK_FAMILY="$2"
shift 2  # everything else goes straight to aws ecs run-task

DEFAULT_SUBNETS=$(aws ec2 describe-subnets \
  --filters "Name=default-for-az,Values=true" \
  --region eu-central-1 --profile weather \
  --query "Subnets[].SubnetId" --output text)

if [ -z "$DEFAULT_SUBNETS" ]; then
  echo "No subnets found for ECS tasks" >&2
  exit 1
fi

SUBNETS_COMMA=$(echo "$DEFAULT_SUBNETS" | tr '\t' ',')

echo "Running $TASK_FAMILY in cluster $CLUSTER ..."
aws ecs run-task \
  --cluster "$CLUSTER" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS_COMMA],assignPublicIp=ENABLED}" \
  --task-definition "$TASK_FAMILY" \
  "$@"
