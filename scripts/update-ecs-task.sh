#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <task-json> <cluster-name> [service-name]" >&2
  exit 1
fi

TASK_JSON="$1"
CLUSTER="$2"
SERVICE="${3:-}"

echo "Registering new task definition from $TASK_JSON ..."
REVISION_ARN=$(aws ecs register-task-definition \
  --cli-input-json file://"$TASK_JSON" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

echo "Registered new revision: $REVISION_ARN"

if [ -n "$SERVICE" ]; then
  echo "Updating service $SERVICE in cluster $CLUSTER to use new revision ..."
  aws ecs update-service \
    --cluster "$CLUSTER" \
    --service "$SERVICE" \
    --task-definition "$REVISION_ARN"
  echo "Service $SERVICE updated to use $REVISION_ARN"
else
  echo "No service name provided. Task definition registered, but no service updated."
fi
