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


output "ecs_cluster_arn" {
    description = "ECS weather-cluster ARN"
    value = aws_ecs_cluster.weather.arn
}
