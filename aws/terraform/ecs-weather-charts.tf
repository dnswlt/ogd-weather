# Task definition
resource "aws_ecs_task_definition" "weather_charts" {
  family                   = "weather-charts"
  cpu                      = "512"
  memory                   = "1024"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = data.aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.weather_task.arn

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
                    "value": "/tmp"
                },
                {
                    "name": "OGD_DB_HOST",
                    "value": "${aws_db_instance.postgres.address}"
                },
                {
                    "name": "OGD_DB_PORT",
                    "value": "${aws_db_instance.postgres.port}"
                },
                {
                    "name": "OGD_DB_DBNAME",
                    "value": "${aws_db_instance.postgres.db_name}"
                }
            ],
            "secrets": [
                {
                    "name": "OGD_POSTGRES_ROLE_SECRET",
                    "valueFrom": "${data.aws_secretsmanager_secret.db_credentials_app_role.arn}"
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
