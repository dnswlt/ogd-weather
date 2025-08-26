# Event Bridge schedule to run the db_updater task daily.
resource "aws_scheduler_schedule" "weather_db_updater" {
  name = "daily-weather-db-updater"

  flexible_time_window {
    mode = "FLEXIBLE"
    maximum_window_in_minutes = 15
  }

  schedule_expression = "cron(50 15 * * ? *)" # Runs at 15:50 UTC daily

  target {
    arn      = aws_ecs_cluster.weather.arn
    role_arn = aws_iam_role.weather_db_updater_scheduler_role.arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.weather_db_updater.arn
      launch_type         = "FARGATE"

      network_configuration {
        subnets          = [aws_subnet.subnet_a.id, aws_subnet.subnet_b.id, aws_subnet.subnet_c.id]
        security_groups  = [aws_security_group.ecs_tasks.id]
        assign_public_ip = true
      }
    }
  }
}

resource "aws_iam_role" "weather_db_updater_scheduler_role" {
  name = "daily-weather-db-updater-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = { Service = "scheduler.amazonaws.com" },
        Action = "sts:AssumeRole"
      }
    ]
  })
}


resource "aws_iam_policy" "weather_db_updater_scheduler_policy" {
  name = "daily-weather-db-updater-policy"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = "ecs:RunTask",
        Resource = aws_ecs_task_definition.weather_db_updater.arn
        Condition = {
          ArnEquals = { "ecs:cluster" = aws_ecs_cluster.weather.arn }
        }
      },
      {
        Effect = "Allow",
        Action = "iam:PassRole",
        Resource = [
          aws_ecs_task_definition.weather_db_updater.task_role_arn,
          aws_ecs_task_definition.weather_db_updater.execution_role_arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "weather_db_updater_scheduler_attachment" {
  role       = aws_iam_role.weather_db_updater_scheduler_role.name
  policy_arn = aws_iam_policy.weather_db_updater_scheduler_policy.arn
}
