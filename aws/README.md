# AWS

We configure AWS assets with Terraform (infrastructure as code, *IaC*).

## Initial setup steps

Secrets and the S3 bucket to store Terraform's own state are *not*
defined in Terraform.

So besides the usual `terraform plan/apply` cycle inside `./terraform`,
the following steps are required:

1. `./create_tf_assets.sh` to create the S3 bucket and DynamoDB for Terraform
   lock management.

2. `./create_db_secrets.sh` to create the password secret for the Postgres DB app user.

