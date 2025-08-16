# Definitions of ECR repositories and policies.

resource "aws_ecr_repository" "weather_api" {
  name                 = "weather-api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration     { encryption_type = "AES256" }
  tags = { project = "weather", iac = "terraform" }
}

resource "aws_ecr_repository" "weather_charts" {
  name                 = "weather-charts"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration     { encryption_type = "AES256" }
  tags = { project = "weather", iac = "terraform" }
}

locals {
  weather_ecr_policy = <<JSON
    {
        "rules": [
            {
                "rulePriority": 10,
                "description": "Expire latest images",
                "selection": {
                    "tagStatus": "tagged",
                    "tagPrefixList": ["latest"],
                    "countType": "imageCountMoreThan",
                    "countNumber": 10
                },
                "action": { "type": "expire" }
            },
            {
                "rulePriority": 20,
                "description": "Expire versioned images",
                "selection": {
                    "tagStatus": "tagged",
                    "tagPrefixList": ["v"],
                    "countType": "imageCountMoreThan",
                    "countNumber": 10
                },
                "action": { "type": "expire" }
            },
            {
                "rulePriority": 30,
                "description": "Expire untagged images",
                "selection": {
                    "tagStatus": "untagged",
                    "countType": "imageCountMoreThan",
                    "countNumber": 5
                },
                "action": { "type": "expire" }
            }
        ]
    }
JSON
}

resource "aws_ecr_lifecycle_policy" "weather_api_policy" {
  repository = aws_ecr_repository.weather_api.name
  policy     = local.weather_ecr_policy
}

resource "aws_ecr_lifecycle_policy" "weather_charts_policy" {
  repository = aws_ecr_repository.weather_charts.name
  policy     = local.weather_ecr_policy
}
