#!/usr/bin/env bash
set -euo pipefail

CLUSTER="weather-cluster"

# Get all running tasks
TASKS=$(aws ecs list-tasks \
  --cluster "$CLUSTER" \
  --region "eu-central-1" \
  --profile "weather" \
  --desired-status RUNNING \
  --query 'taskArns' \
  --output text)

if [[ -z "$TASKS" ]]; then
  echo "No running tasks in cluster $CLUSTER"
  exit 0
fi

aws ecs describe-tasks \
  --cluster "$CLUSTER" \
  --tasks "$TASKS" \
  --query 'tasks[].{
    TaskId: split(taskArn, `/`)[-1],
    Definition: split(taskDefinitionArn, `/`)[-1],
    LastStatus: lastStatus,
    Desired: desiredStatus,
    Containers: containers[].lastStatus
  }' \
  --output table
