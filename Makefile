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

CLUSTER = weather-cluster

.PHONY: build rebuild up down restart update-db recreate-views logs clean
.PHONY: aws-login aws-build aws-push aws-redeploy aws-roll aws-update-db

## Local testing =====================================================
.PHONY: test test-api test-db-updater test-charts test-integration

# --- General Test Commands ---

# Runs all tests sequentially.
test: test-api test-db-updater test-charts

test-api:
	@echo "--- Running tests for Go service/api ---"
	cd service/api && go test ./...

test-db-updater:
	@echo "--- Running tests for Python service/db_updater ---"
	pytest service/db_updater

test-charts:
	@echo "--- Running tests for Python service/charts ---"
	pytest service/charts

test-integration:
	@echo "--- Running integration tests for Python service/db_updater ---"
	pytest -m integration -o log_cli=true --log-cli-level=INFO service/db_updater


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
	$(COMPOSE) down && $(COMPOSE) up -d api charts

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
	bash aws/scripts/aws_build.sh

aws-push: ## Push all images to ECR
	bash aws/scripts/aws_push.sh

aws-redeploy: ## Roll charts first (wait), then API
	$(MAKE) aws-roll SERVICE=weather-charts-service
	$(MAKE) aws-roll SERVICE=weather-api-service

aws-roll: ## Usage: make roll SERVICE=weather-api-service
	aws ecs update-service --no-cli-pager --cluster $(CLUSTER) --service $(SERVICE) --force-new-deployment
	aws ecs wait services-stable --cluster $(CLUSTER) --services $(SERVICE)
	@echo "Rolled $(SERVICE) on $(CLUSTER)"

aws-update-db:
	bash aws/scripts/run_db_updater.sh

aws-recreate-db:
	bash aws/scripts/run_db_updater_overrides.sh --force-recreate

aws-terraform-apply:
	cd aws/terraform && terraform apply
