# Role assumed by the ECS agent to pull images, write logs, fetch secrets.
# (Data source because we use the AWS-provided standard role.)
data "aws_iam_role" "ecs_task_execution" {
  name = "ecsTaskExecutionRole"
}

# Role assumed by the application containers to call AWS services.
resource "aws_iam_role" "weather_task" {
  name = "weather-task-role"

  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Effect    = "Allow",
        Principal = { Service = "ecs-tasks.amazonaws.com" },
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

# Single managed policy for all runtime config reads (Secrets + SSM).
# (Typically injected into tasks using environment variables.)
resource "aws_iam_policy" "ecs_exec_runtime_config_read" {
  name        = "ecs-exec-read-weather-runtime-config"
  description = "Allow ECS task execution role to read Secrets Manager secrets and SSM parameters for weather services"

  policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Sid      = "SecretsManagerRead",
        Effect   = "Allow",
        Action   = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"],
        Resource = [
          aws_db_instance.postgres.master_user_secret[0].secret_arn,
          data.aws_secretsmanager_secret.db_credentials_app_role.arn
        ]
      },
      {
        Sid      = "SSMParameterRead",
        Effect   = "Allow",
        Action   = ["ssm:GetParameter", "ssm:GetParameters"],
        Resource = data.aws_ssm_parameter.api_bearer_token.arn
      }
    ]
  })
}

# Attach the managed policy to the ECS task execution role
resource "aws_iam_role_policy_attachment" "ecs_exec_attach_runtime_config_read" {
  role       = data.aws_iam_role.ecs_task_execution.name
  policy_arn = aws_iam_policy.ecs_exec_runtime_config_read.arn
}

