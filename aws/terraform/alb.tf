
resource "aws_lb" "api" {
  name               = "weather-alb"
  load_balancer_type = "application"
  internal           = false
  security_groups    = [aws_security_group.alb.id]
  subnets            = [
    aws_subnet.subnet_a.id,
    aws_subnet.subnet_b.id,
    aws_subnet.subnet_c.id 
  ]

  ip_address_type = "ipv4"

  tags = { Name = "weather-alb" }
}

resource "aws_lb_target_group" "api" {
  name        = "weather-api-tg"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  # Matches your TG
  port     = 8080
  protocol = "HTTP"

  health_check {
    enabled             = true
    protocol            = "HTTP"
    path                = "/health"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 5
    unhealthy_threshold = 2
    port                = "traffic-port"
  }

  protocol_version = "HTTP1"
  ip_address_type  = "ipv4"

  tags = { Name = "weather-api-tg" }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"
  certificate_arn   = "arn:aws:acm:eu-central-1:006725292903:certificate/843e52ba-ad93-4005-a006-a2c5f7bbe89c"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}
