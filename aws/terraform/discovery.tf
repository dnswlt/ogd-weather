# Service Discovery

resource "aws_service_discovery_private_dns_namespace" "weather" {
  name = "weather.internal" # avoid using .local (DNS resolver treats *.local as mDNS, per RFC 6762)
  vpc  = aws_vpc.main.id
}

# Charts discovery service in that namespace
resource "aws_service_discovery_service" "weather_charts" {
  name = "weather-charts"

  dns_config {
    namespace_id   = aws_service_discovery_private_dns_namespace.weather.id
    routing_policy = "MULTIVALUE"
    dns_records {
      type = "A"
      ttl  = 5
    }
  }
}
