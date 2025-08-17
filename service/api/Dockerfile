# --- build stage ---
FROM golang:1.24-alpine AS builder

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILDTIME=unknown

WORKDIR /app

# Install JS toolchain
RUN apk add --no-cache nodejs npm

# Go deps
COPY go.mod ./
RUN go mod download

# Copy everything inside build context: ./service/api
COPY . .

# Build frontend (web -> static/dist)
WORKDIR /app/web
RUN npm install && npm run build

# Back to Go build
WORKDIR /app
RUN CGO_ENABLED=0 go build \
    -ldflags "\
        -X 'github.com/dnswlt/ogd-weather/service/api/internal/version.Version=${VERSION}' \
        -X 'github.com/dnswlt/ogd-weather/service/api/internal/version.GitCommit=${COMMIT}' \
        -X 'github.com/dnswlt/ogd-weather/service/api/internal/version.BuildTime=${BUILDTIME}'" \
    -o weather-api ./cmd/server

# --- runtime stage ---
FROM alpine:3.20

# Add deps:
#   * curl for health checks in the cloud
#   * tzdata for time zone information used by the Go server
RUN apk add --no-cache curl tzdata

WORKDIR /app

COPY --from=builder /app/weather-api /app/
COPY --from=builder /app/templates /app/templates
COPY --from=builder /app/static /app/static

EXPOSE 8080

# Ensure Go app finds time zone infos for "Europe/Zurich" etc.
ENV ZONEINFO=/usr/share/zoneinfo

CMD ["/app/weather-api", "-addr", "0.0.0.0:8080"]
