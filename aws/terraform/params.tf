# SSM parameters

# Bearer token for /status page access.
data "aws_ssm_parameter" "api_bearer_token" {
  name = "/weather/api/BearerToken"
}
