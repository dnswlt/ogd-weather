
# This secret is defined outside of Terraform.
# Define it here as a data entity for convenient access.
data "aws_secretsmanager_secret" "db_credentials_app_role" {
  name = "/weather/db/credentials/app_role"
}
