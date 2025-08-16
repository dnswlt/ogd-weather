
# Role assumed by the ECS agent to pull images, write logs, fetch secrets.
# Just a data ref., not a resource, b/c we're using the standard AWS role.
data "aws_iam_role" "ecs_task_execution" {
  name = "ecsTaskExecutionRole"
}

# Role assumed by the application containers to call AWS services.
resource "aws_iam_role" "weather_task" {
  name = "weather-task-role"

  assume_role_policy = <<JSON
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": { 
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
JSON
}
