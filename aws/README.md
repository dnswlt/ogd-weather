# AWS

We configure AWS assets with Terraform (infrastructure as code, *IaC*).

## Initial setup steps

Secrets and the S3 bucket to store Terraform's own state are *not*
defined in Terraform.

So besides the usual `terraform plan/apply` cycle inside `./terraform`,
the following steps are required:

1. [./scripts/create_tf_assets.sh](./scripts/create_tf_assets.sh)
   to create the S3 bucket for Terraform state management.

2. [./scripts/create_secrets_db.sh](./scripts/create_secrets_db.sh)
   to create the password secret for the Postgres DB app user.

3. [./scripts/create_secrets_bearer.sh](./scripts/create_secrets_bearer.sh)
   to create the BearerToken for the `api` service's admin endpoints.


## Releases

We use UTC timestamps and git commit hashes to tag Docker images in AWS ECR.
The format is `v{YYYYmmdd}_${HHMM}_{hash}`, for example: `v20250818_0718_a6763ba`.

The latest tag values are stored as AWS SSM parameters, which are used by
Terraform to identify the versions to be deployed.

To build and deploy a new release, make sure you are in a clean repo state and run
the following commands on a Linux machine:

```bash
make aws-build

make aws-push
```

## Terraform

Make sure instantiate your local `terraform.tfvars` from the given example
file first (the actual tfvars are not part of the Git repo):

```bash
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
# Adjust variables:
vim terraform/terraform.tfvars
```

Otherwise nothing special here, just the regular Terraform cycle:

```bash
cd ./terraform

aws sso login --profile ${AWS_PROFILE}

terraform plan

terraform apply
```
