#!/usr/bin/env bash
set -euo pipefail

# Runs the db-updater task with arguments passed to the script.

# Check if at least one argument is provided for the command.
if [ "$#" -eq 0 ]; then
  echo "Error: No command arguments provided." >&2
  echo "Usage: $0 <arg1> <arg2> ..." >&2
  exit 1
fi

# cd to the terraform directory for "terraform output" calls.
cd $(dirname $0)/../terraform

echo "Collecting terraform outputs."

SUBNET_IDS=$(terraform output -json ecs_subnet_ids)
SECURITY_GROUP=$(terraform output -json security_group_ecs_tasks)
TASK_DEF=$(terraform output -raw weather_db_updater_task_def_arn)
CLUSTER=$(terraform output -raw ecs_cluster_arn)

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

# Build JSON list from script's arguments.
COMMAND_ARGS_JSON=$(jq -n '$ARGS.positional' --args -- "$@")

COMMAND_OVERRIDES=$( \
  jq -n \
    --argjson command_args "$COMMAND_ARGS_JSON" \
    '{
      "containerOverrides": [
        {
          "name": "weather-db-updater",
          "command": $command_args
        }
      ]
    }'
)


echo "Starting task $TASK_DEF in cluster $CLUSTER with command: $@"

aws ecs run-task \
  --no-cli-pager \
  --cluster "$CLUSTER" \
  --task-definition "$TASK_DEF" \
  --launch-type FARGATE \
  --count 1 \
  --network-configuration "$NETWORK_CONFIG" \
  --overrides "$COMMAND_OVERRIDES"
