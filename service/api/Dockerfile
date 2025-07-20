# --- build stage ---
FROM golang:1.24-alpine AS builder

WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o weather-api ./cmd/server

# --- runtime stage ---
FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app/weather-api /app/
COPY templates/ templates/

# Expose the port your Go service listens on
EXPOSE 8080

CMD ["/app/weather-api", "-addr", "0.0.0.0:8080"]

