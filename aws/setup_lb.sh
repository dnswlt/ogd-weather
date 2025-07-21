CERT_ARN="arn:aws:acm:eu-central-1:006725292903:certificate/843e52ba-ad93-4005-a006-a2c5f7bbe89c"

# === Get default networking bits ===
DEFAULT_VPC=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --region eu-central-1 --profile weather \
  --query "Vpcs[0].VpcId" --output text)

DEFAULT_SUBNETS=$(aws ec2 describe-subnets \
  --filters "Name=default-for-az,Values=true" \
  --region eu-central-1 --profile weather \
  --query "Subnets[].SubnetId" --output text)

DEFAULT_SG=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=default" \
  --region eu-central-1 --profile weather \
  --query "SecurityGroups[0].GroupId" --output text)

echo "Default VPC: $DEFAULT_VPC"
echo "Default subnets: $DEFAULT_SUBNETS"
echo "Default SG: $DEFAULT_SG"

# === Create ALB ===
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name weather-alb \
  --type application \
  --subnets $DEFAULT_SUBNETS \
  --security-groups $DEFAULT_SG \
  --region eu-central-1 --profile weather \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

echo "ALB ARN: $ALB_ARN"

# === Create 2 Target Groups ===
TG_API=$(aws elbv2 create-target-group \
  --name weather-api-tg \
  --protocol HTTP --port 8080 \
  --vpc-id $DEFAULT_VPC \
  --target-type ip \
  --region eu-central-1 --profile weather \
  --query 'TargetGroups[0].TargetGroupArn' --output text)

TG_CHARTS=$(aws elbv2 create-target-group \
  --name weather-charts-tg \
  --protocol HTTP --port 8080 \
  --vpc-id $DEFAULT_VPC \
  --target-type ip \
  --region eu-central-1 --profile weather \
  --query 'TargetGroups[0].TargetGroupArn' --output text)

echo "API Target Group: $TG_API"
echo "Charts Target Group: $TG_CHARTS"

# === Create HTTPS Listener with default 404 ===
LISTENER_ARN=$(aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTPS --port 443 \
  --certificates CertificateArn="$CERT_ARN" \
  --default-actions Type=fixed-response,FixedResponseConfig="{StatusCode=404,ContentType=text/plain,MessageBody='Not Found'}" \
  --region eu-central-1 --profile weather \
  --query 'Listeners[0].ListenerArn' --output text)

echo "HTTPS Listener: $LISTENER_ARN"

# === Add path rules ===
aws elbv2 create-rule \
  --listener-arn $LISTENER_ARN \
  --priority 10 \
  --conditions Field=path-pattern,Values='/api/*' \
  --actions Type=forward,TargetGroupArn=$TG_API \
  --region eu-central-1 --profile weather

aws elbv2 create-rule \
  --listener-arn $LISTENER_ARN \
  --priority 20 \
  --conditions Field=path-pattern,Values='/charts/*' \
  --actions Type=forward,TargetGroupArn=$TG_CHARTS \
  --region eu-central-1 --profile weather
