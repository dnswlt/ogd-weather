# Security groups

# Default SG (must exist, can be narrowed down later).
resource "aws_security_group" "default" {
  name        = "default"
  description = "default VPC security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "Allow all traffic from self"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  ingress {
    description = "Allow HTTPS from anywhere"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "default-sg"
  }
}

# Public-facing ALB: allow 80/443 from the internet; all egress.
resource "aws_security_group" "alb" {
  name        = "weather-alb-sg"
  description = "ALB ingress 80/443 from internet"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    protocol    = "tcp"
    from_port   = 80
    to_port     = 80
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All egress"
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "weather-alb-sg" }
}

# ECS tasks: no public ingress; allow traffic ONLY from the ALB on app port.
# Full egress so tasks can call RDS/EFS/Internet.
resource "aws_security_group" "ecs_tasks" {
  name        = "weather-ecs-tasks-sg"
  description = "Ingress from ALB on app port; full egress"
  vpc_id      = aws_vpc.main.id

  # Allow all ECS tasks to talk to each other.
  ingress {
    description = "Allow all traffic from self"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  # ALB -> tasks on container port 8080 (adjust if your API uses a different port)
  ingress {
    description     = "ALB to ECS tasks"
    protocol        = "tcp"
    from_port       = 8080
    to_port         = 8080
    security_groups = [aws_security_group.alb.id] # source = ALB SG
  }

  egress {
    description = "All egress"
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "weather-ecs-tasks-sg" }
}

# RDS Postgres: allow only from ECS tasks on 5432.
resource "aws_security_group" "db" {
  name        = "weather-db-sg"
  description = "Postgres inbound from ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Postgres from ECS tasks"
    protocol        = "tcp"
    from_port       = 5432
    to_port         = 5432
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  egress {
    description = "All egress"
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "weather-db-sg" }
}

output "security_group_ecs_tasks" {
  description = "Subnets to run the ECS tasks in"
  value       = aws_security_group.ecs_tasks.id
}
