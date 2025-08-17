#!/usr/bin/env bash
set -euo pipefail

# Runs the db-updater task to update weather data CSVs.

# cd to the terraform directory for "terraform output" calls.
cd $(dirname $0)/../terraform

# Runs the db-updater task with --bootstrap-postgres

echo "Collecting terraform outputs."

SUBNET_IDS=$(terraform output -json ecs_subnet_ids)
SECURITY_GROUP=$(terraform output -json security_group_ecs_tasks)

NETWORK_CONFIG=$(cat <<JSON
{
  "awsvpcConfiguration": {
    "subnets": $SUBNET_IDS,
    "securityGroups": [$SECURITY_GROUP],
    "assignPublicIp": "ENABLED"
  }
}
JSON
)

TASK_DEF=$(terraform output -raw weather_db_updater_task_def_arn)

CLUSTER=$(terraform output -raw ecs_cluster_arn)

echo "Starting task $TASK_DEF in cluster $CLUSTER."

aws ecs run-task \
  --cluster "$CLUSTER" \
  --task-definition "$TASK_DEF" \
  --launch-type FARGATE \
  --count 1 \
  --network-configuration "$NETWORK_CONFIG"
