# --- build stage ---
FROM golang:1.24-alpine AS builder

# Build-time args (can be passed by docker build --build-arg)
ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILDTIME=unknown

WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build \
    -ldflags "\
        -X 'github.com/dnswlt/ogd-weather/service/api/internal/version.Version=${VERSION}' \
        -X 'github.com/dnswlt/ogd-weather/service/api/internal/version.GitCommit=${COMMIT}' \
        -X 'github.com/dnswlt/ogd-weather/service/api/internal/version.BuildTime=${BUILDTIME}'" \
    -o weather-api ./cmd/server

# --- runtime stage ---
FROM alpine:3.20
RUN apk add --no-cache tzdata
WORKDIR /app
COPY --from=builder /app/weather-api /app/
COPY templates ./templates
COPY static ./static

# Expose the port your Go service listens on
EXPOSE 8080
# Ensure Go app finds time zone infos for "Europe/Zurich" etc.
ENV ZONEINFO=/usr/share/zoneinfo

CMD ["/app/weather-api", "-addr", "0.0.0.0:8080"]

