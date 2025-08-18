# Definitions of ECR repositories and policies.

# Repositories

resource "aws_ecr_repository" "weather_api" {
  name                 = "weather-api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
  tags = { project = "weather", iac = "terraform" }
}

resource "aws_ecr_repository" "weather_charts" {
  name                 = "weather-charts"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
  tags = { project = "weather", iac = "terraform" }
}

resource "aws_ecr_repository" "weather_db_updater" {
  name                 = "weather-db-updater"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
  tags = { project = "weather", iac = "terraform" }
}

# Images and tags of currently deployed images

data "aws_ssm_parameter" "weather_api_tag" {
  name = "/weather/images/api/tag"
}

data "aws_ssm_parameter" "weather_charts_tag" {
  name = "/weather/images/charts/tag"
}

data "aws_ssm_parameter" "weather_db_updater_tag" {
  name = "/weather/images/db_updater/tag"
}

data "aws_ecr_image" "api" {
  repository_name = aws_ecr_repository.weather_api.name
  image_tag       = data.aws_ssm_parameter.weather_api_tag.value
}

data "aws_ecr_image" "charts" {
  repository_name = aws_ecr_repository.weather_charts.name
  image_tag       = data.aws_ssm_parameter.weather_charts_tag.value
}

data "aws_ecr_image" "db_updater" {
  repository_name = aws_ecr_repository.weather_db_updater.name
  image_tag       = data.aws_ssm_parameter.weather_db_updater_tag.value
}

# Policies

locals {
  weather_ecr_policy = jsonencode({
    rules = [
      {
        rulePriority = 10
        description  = "Expire latest images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["latest"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 20
        description  = "Expire untagged images"
        selection = {
          tagStatus   = "untagged"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 99
        description  = "Catchall to expire commit-hash versioned images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}


resource "aws_ecr_lifecycle_policy" "weather_api_policy" {
  repository = aws_ecr_repository.weather_api.name
  policy     = local.weather_ecr_policy
}

resource "aws_ecr_lifecycle_policy" "weather_charts_policy" {
  repository = aws_ecr_repository.weather_charts.name
  policy     = local.weather_ecr_policy
}

resource "aws_ecr_lifecycle_policy" "weather_db_updater_policy" {
  repository = aws_ecr_repository.weather_db_updater.name
  policy     = local.weather_ecr_policy
}
