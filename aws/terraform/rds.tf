# Subnet group for RDS (uses your 3 existing subnets)
resource "aws_db_subnet_group" "weather" {
  name       = "weather-db-subnets"
  subnet_ids = [
    aws_subnet.subnet_a.id,
    aws_subnet.subnet_b.id,
    aws_subnet.subnet_c.id
  ]
  tags = { Name = "weather-db-subnets" }
}

# Private PostgreSQL instance (hosted)
resource "aws_db_instance" "postgres" {
  identifier                  = "weather-postgres"
  engine                      = "postgres"
  engine_version              = "17"                # latest major supported; minor auto-upgrades on
  instance_class              = "db.t4g.micro"      # start small, can resize later
  allocated_storage           = 20                  # GB to start
  max_allocated_storage       = 100                 # storage autoscaling up
  storage_type                = "gp3"
  storage_encrypted           = true

  db_subnet_group_name        = aws_db_subnet_group.weather.name
  vpc_security_group_ids      = [aws_security_group.db.id]
  publicly_accessible         = false               # no public IP
  multi_az                    = false               # enable later if you want HA
  backup_retention_period     = 7
  auto_minor_version_upgrade  = true
  deletion_protection         = false               # set true later if you want a guardrail

  # Master user (password stored in Secrets Manager by AWS)
  username                    = "weather_admin"
  manage_master_user_password = true

  # Create the initial (and only) DB.
  db_name = "weather"

  tags = { Name = "weather-postgres" }
}

# Handy outputs
output "postgres_endpoint" {
  value = aws_db_instance.postgres.address
}

# Secrets Manager ARN for the master password (created by RDS)
output "postgres_master_secret_arn" {
  # master_user_secret can be null briefly during creation; guard with try()
  value = try(aws_db_instance.postgres.master_user_secret[0].secret_arn, null)
}
