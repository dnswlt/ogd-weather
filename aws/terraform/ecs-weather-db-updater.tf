# Task definition
resource "aws_ecs_task_definition" "weather_db_updater" {
  family                   = "weather-db-updater"
  cpu                      = "1024"
  memory                   = "4096" # Needs ~ 2GiB during view creation
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = data.aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.weather_task.arn

  container_definitions = jsonencode([
    {
      name         = "weather-db-updater"
      image        = "${aws_ecr_repository.weather_db_updater.repository_url}@${data.aws_ecr_image.db_updater.image_digest}"
      cpu          = 0
      portMappings = []
      essential    = true
      environment = [
        {
          name  = "OGD_BASE_DIR"
          value = "/tmp"
        },
        {
          name  = "OGD_DB_HOST"
          value = aws_db_instance.postgres.address
        },
        {
          name  = "OGD_DB_PORT"
          value = tostring(aws_db_instance.postgres.port)
        },
        {
          name  = "OGD_DB_DBNAME"
          value = aws_db_instance.postgres.db_name
        }
      ]
      secrets = [
        {
          name      = "OGD_POSTGRES_MASTER_SECRET"
          valueFrom = aws_db_instance.postgres.master_user_secret[0].secret_arn
        },
        {
          name      = "OGD_POSTGRES_ROLE_SECRET"
          valueFrom = data.aws_secretsmanager_secret.db_credentials_app_role.arn
        }
      ]
      volumesFrom = []
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/weather-db-updater"
          awslogs-region        = "eu-central-1"
          awslogs-stream-prefix = "ecs"
        }
      }
      systemControls = []
    }
  ])
}


output "weather_db_updater_task_def_arn" {
  value = aws_ecs_task_definition.weather_db_updater.arn
}
