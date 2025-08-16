# ECS Cluster
resource "aws_ecs_cluster" "weather" {
  name = "weather-cluster"

  setting {
    name  = "containerInsights"
    value = "disabled"
  }

  tags = {
    Name = "weather-cluster"
  }
}
