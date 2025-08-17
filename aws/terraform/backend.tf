terraform {
  backend "s3" {
    bucket       = "ogd-weather-tfstate-006725292903-eu-central-1"
    key          = "aws/terraform/terraform.tfstate"
    region       = "eu-central-1"
    use_lockfile = true
    encrypt      = true
  }
}

