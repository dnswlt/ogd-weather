# Task definition
resource "aws_ecs_task_definition" "weather_charts" {
  family                   = "weather-charts"
  cpu                      = "512"
  memory                   = "1024"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = data.aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.weather_task.arn

  # --- VOLUMES (fill from jq output) ---
  # Example (replace IDs/flags to exactly match your JSON):
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

  # Container definitions: paste EXACT JSON from your dump to avoid drift
  container_definitions = <<JSON
    [
        {
            "name": "weather-charts",
            "image": "006725292903.dkr.ecr.eu-central-1.amazonaws.com/weather-charts:${var.weather_charts_version}",
            "cpu": 0,
            "portMappings": [
                {
                    "containerPort": 8080,
                    "hostPort": 8080,
                    "protocol": "tcp"
                }
            ],
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
                    "awslogs-group": "/ecs/weather-charts",
                    "awslogs-region": "eu-central-1",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "systemControls": []
        }
    ]
JSON
}

resource "aws_ecs_service" "weather_charts" {
  name            = "weather-charts-service"
  cluster         = aws_ecs_cluster.weather.id
  task_definition = aws_ecs_task_definition.weather_charts.arn
  desired_count   = 0
  launch_type     = "FARGATE"
  enable_execute_command = true


  service_registries {
    registry_arn = aws_service_discovery_service.weather_charts.arn
  }

  network_configuration {
    subnets = [
      aws_subnet.subnet_a.id,
      aws_subnet.subnet_b.id,
      aws_subnet.subnet_c.id
    ]
    security_groups  = [aws_security_group.default.id]
    assign_public_ip = true
  }

  # No load_balancer{} block on purpose (service isn't attached to ALB)

  tags = { Name = "weather-charts-service" }
}
