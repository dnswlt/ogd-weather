# Task definition
resource "aws_ecs_task_definition" "weather_db_updater" {
  family                   = "weather-db-updater"
  cpu                      = "512"
  memory                   = "1024"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn = data.aws_iam_role.ecs_task_execution.arn
  task_role_arn = aws_iam_role.weather_task.arn

  # EFS volume (match existing exactly)
  volume {
    name                = "weather-sqlite"
    configure_at_launch = false

    efs_volume_configuration {
      file_system_id          = "fs-05754135c68678c1d"
      root_directory          = "/"
      transit_encryption      = "ENABLED"
      transit_encryption_port = 0

      authorization_config {
        access_point_id = "fsap-0e0734cec27f5a6df"
        iam             = "DISABLED"
      }
    }
  }

  container_definitions = <<JSON
    [  
        {
            "name": "weather-db-updater",
            "image": "006725292903.dkr.ecr.eu-central-1.amazonaws.com/weather-db-updater:${var.weather_db_updater_version}",
            "cpu": 0,
            "portMappings": [],
            "essential": true,
            "environment": [
                { 
                    "name": "OGD_BASE_DIR",
                    "value": "/app/efs"
                }
            ],
            "mountPoints": [
                { 
                    "sourceVolume": "weather-sqlite",
                    "containerPath": "/app/efs",
                    "readOnly": false
                }
            ],
            "volumesFrom": [],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/weather-db-updater",
                    "awslogs-region": "eu-central-1",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "systemControls": []
        }
    ]
JSON
}
