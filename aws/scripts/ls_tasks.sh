#!/usr/bin/env bash
set -euo pipefail

CLUSTER="weather-cluster"

aws ecs list-tasks \
  --cluster "${CLUSTER}" \
  --desired-status RUNNING \
  --query 'taskArns' \
  --output text \
| xargs aws ecs describe-tasks \
    --cluster "${CLUSTER}" \
    --tasks \
| jq '
  .tasks[] |
  {
    id: (.taskArn | split("/")[-1]),
    service: (.group | sub("^service:";"")),
    lastStatus,
    desiredStatus,
    startedAt,
    image: (.containers[0].image | split("/")[-1]),
    taskDefinitionArn
  }
'
