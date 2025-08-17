# Had some weird docker "Permission denied" errors without it.
SHELL := /bin/bash

VERSION    ?= $(shell git describe --tags --always 2>/dev/null || echo dev)
COMMIT     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILDTIME  ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

# Default DB is postgres. Switch with: `make <command> DB=sqlite`
ifeq ($(DB),sqlite)
  COMPOSE := docker compose -f compose.yml
else
  COMPOSE := docker compose -f compose.yml -f compose.pg.yml --env-file .env
endif

# AWS settings (must be provided via environment or .env file)
AWS_ACCOUNT_ID ?= ACCOUNT_ID_NOT_SET
AWS_REGION ?= $(if $(AWS_DEFAULT_REGION),$(AWS_DEFAULT_REGION),AWS_REGION_NOT_SET)
ECR_REGISTRY = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

CHARTS_IMAGE = $(ECR_REGISTRY)/weather-charts:latest
API_IMAGE = $(ECR_REGISTRY)/weather-api:latest
DB_UPDATER_IMAGE = $(ECR_REGISTRY)/weather-db-updater:latest

CLUSTER = weather-cluster

.PHONY: build rebuild up down restart update-db recreate-views logs clean
.PHONY: aws-login aws-build aws-push aws-redeploy aws-roll aws-update-db

## Local testing =====================================================
.PHONY: test test-api test-db-updater test-charts

# --- General Test Commands ---

# Runs all tests sequentially.
test: test-api test-db-updater test-charts

test-api:
	@echo "--- Running tests for Go service/api ---"
	cd service/api && go test ./...

test-db-updater:
	@echo "--- Running tests for Python service/db_updater ---"
	cd service/db_updater && pytest

test-charts:
	@echo "--- Running tests for Python service/charts ---"
	cd service/charts && pytest


## Local Dev =========================================================

build: ## Build all local images
	$(COMPOSE) build

rebuild: ## Force rebuild of all local images
	$(COMPOSE) build --no-cache

up: ## Start api + charts WITHOUT rebuilding
	$(COMPOSE) up -d api charts

down: ## Stop all running containers (but keep images & volumes)
	$(COMPOSE) down

restart: ## Restart api + charts containers without rebuild
	$(COMPOSE) down && $(COMPOSE) up -d

update-db: ## Run db-updater batch job WITHOUT rebuilding
	$(COMPOSE) run --rm db-updater

recreate-views:
	$(COMPOSE) run --rm db-updater --recreate-views

logs: ## Follow logs for api + charts
	$(COMPOSE) logs -f api charts

clean: ## Stop everything and remove containers, images, volumes
	$(COMPOSE) down --rmi local --volumes --remove-orphans

## AWS Build & Deploy ===============================================

aws-login: ## Authenticate Docker with ECR
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_REGISTRY)

aws-build: ## Build AWS images (charts, api, db-updater)
	docker build -t $(CHARTS_IMAGE) -f service/charts/Dockerfile .
	docker build \
		--build-arg VERSION=$(VERSION) \
	  	--build-arg COMMIT=$(COMMIT) \
	  	--build-arg BUILDTIME=$(BUILDTIME) \
	  	-t $(API_IMAGE) -f service/api/Dockerfile ./service/api
	docker build -t $(DB_UPDATER_IMAGE) -f service/db_updater/Dockerfile .

aws-push: ## Push all images to ECR
	docker push $(CHARTS_IMAGE)
	docker push $(API_IMAGE)
	docker push $(DB_UPDATER_IMAGE)

aws-redeploy: ## Roll charts first (wait), then API
	$(MAKE) aws-roll SERVICE=weather-charts-service
	$(MAKE) aws-roll SERVICE=weather-api-service

aws-roll: ## Usage: make roll SERVICE=weather-api-service
	aws ecs update-service --no-cli-pager --cluster $(CLUSTER) --service $(SERVICE) --force-new-deployment
	aws ecs wait services-stable --cluster $(CLUSTER) --services $(SERVICE)
	@echo "Rolled $(SERVICE) on $(CLUSTER)"

aws-update-db:
	bash aws/scripts/run_db_updater.sh
