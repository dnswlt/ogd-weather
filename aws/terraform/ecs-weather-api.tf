# Service and task definition for api service.

resource "aws_ecs_task_definition" "weather_api" {
  family                   = "weather-api"
  cpu                      = "256"
  memory                   = "512"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = data.aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.weather_task.arn

  container_definitions = jsonencode([
    {
      name  = "weather-api"
      image = "${aws_ecr_repository.weather_api.repository_url}@${data.aws_ecr_image.api.image_digest}"
      cpu   = 0
      portMappings = [
        {
          containerPort = 8080
          hostPort      = 8080
          protocol      = "tcp"
        }
      ]
      essential   = true
      environment = [
        {
          name  = "OGD_CHART_SERVICE_ENDPOINT"
          value = "http://weather-charts.weather.internal:8080"
        },
        {
          name  = "OGD_CACHE_SIZE"
          value = "50000000"
        }
      ]
      mountPoints = []
      volumesFrom = []
      secrets = [
        {
          name      = "OGD_BEARER_TOKEN"
          valueFrom = "/weather/api/BearerToken"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/weather-api"
          awslogs-region        = "eu-central-1"
          awslogs-stream-prefix = "ecs"
        }
      }
      systemControls = []
    }
  ])
}


# ECS Service for weather-api-service
resource "aws_ecs_service" "weather_api" {
  name            = "weather-api-service"
  cluster         = aws_ecs_cluster.weather.id
  task_definition = aws_ecs_task_definition.weather_api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets = [
      aws_subnet.subnet_a.id,
      aws_subnet.subnet_b.id,
      aws_subnet.subnet_c.id
    ]
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "weather-api"
    container_port   = 8080
  }

  health_check_grace_period_seconds = 60

  tags = {
    Name = "weather-api-service"
  }
}


output "weather_api_task_def_arn" {
  value = aws_ecs_task_definition.weather_api.arn
}
