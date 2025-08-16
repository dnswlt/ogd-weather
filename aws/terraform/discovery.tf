# Service Discovery

resource "aws_service_discovery_private_dns_namespace" "weather" {
  name = "weather.local"
  vpc  = aws_vpc.main.id
}

resource "aws_service_discovery_service" "weather_charts" {
  name = "weather-charts"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.weather.id

    dns_records {
      type = "A"
      ttl  = 10
    }
  }
}
