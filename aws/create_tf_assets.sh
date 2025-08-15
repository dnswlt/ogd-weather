# Prepare assets for remove terraform state management.
#
# Sets up an S3 bucket (for tfstate) and a DynamoDB (for locking).
#

echo "Creating S3 bucket for tfstate..."

REGION=eu-central-1
BUCKET=ogd-weather-tfstate-006725292903-eu-central-1
LOCK_TABLE=terraform-locks

# S3 bucket
aws s3api create-bucket \
  --bucket "$BUCKET" \
  --region "$REGION" \
  --create-bucket-configuration LocationConstraint="$REGION"

# Versioning (required for safe state)
aws s3api put-bucket-versioning \
  --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled

# Block all public access
aws s3api put-public-access-block \
  --bucket "$BUCKET" \
  --public-access-block-configuration \
'{
  "BlockPublicAcls":true,
  "IgnorePublicAcls":true,
  "BlockPublicPolicy":true,
  "RestrictPublicBuckets":true
}'

# Server-side encryption (SSE-S3)
aws s3api put-bucket-encryption \
  --bucket "$BUCKET" \
  --server-side-encryption-configuration \
'{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

echo "Creating DynamoDB for state locking..."

# DynamoDB table for state locking
aws dynamodb create-table \
  --table-name "$LOCK_TABLE" \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region "$REGION"
