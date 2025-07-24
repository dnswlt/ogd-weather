# Had some weird docker "Permission denied" errors without it.
SHELL := /bin/bash


COMPOSE = docker compose

# AWS settings (must be provided via environment or .env file)
AWS_ACCOUNT_ID ?= ACCOUNT_ID_NOT_SET
AWS_REGION ?= $(if $(AWS_DEFAULT_REGION),$(AWS_DEFAULT_REGION),AWS_REGION_NOT_SET)
ECR_REGISTRY = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

CHARTS_IMAGE = $(ECR_REGISTRY)/weather-charts:latest
API_IMAGE = $(ECR_REGISTRY)/weather-api:latest
DB_UPDATER_IMAGE = $(ECR_REGISTRY)/weather-db-updater:latest

.PHONY: up restart update-db logs clean rebuild aws-login build-aws push-aws deploy-aws

## Local Dev =========================================================

rebuild: ## Force rebuild of all local images
	$(COMPOSE) build --no-cache

up: ## Start api + charts WITHOUT rebuilding
	$(COMPOSE) up -d api charts

down: ## Stop all running containers (but keep images & volumes)
	$(COMPOSE) down

restart: ## Restart api + charts containers without rebuild
	$(COMPOSE) restart api charts

update-db: ## Run db-updater batch job WITHOUT rebuilding
	$(COMPOSE) run --rm db-updater

logs: ## Follow logs for api + charts
	$(COMPOSE) logs -f api charts

clean: ## Stop everything and remove containers, images, volumes
	$(COMPOSE) down --rmi local --volumes --remove-orphans

## AWS Build & Deploy ===============================================

aws-login: ## Authenticate Docker with ECR
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_REGISTRY)

build-aws: ## Build AWS images (charts, api, db-updater)
	docker build -t $(CHARTS_IMAGE) -f service/charts/Dockerfile .
	docker build -t $(API_IMAGE) -f service/api/Dockerfile ./service/api
	docker build -t $(DB_UPDATER_IMAGE) -f service/db_updater/Dockerfile .

push-aws: aws-login build-aws ## Push all images to ECR
	docker push $(CHARTS_IMAGE)
	docker push $(API_IMAGE)
	docker push $(DB_UPDATER_IMAGE)

deploy-aws: ## Deploy updated ECS services (forces new deployment)
	aws ecs update-service --cluster weather-cluster --service weather-charts-service --force-new-deployment
	aws ecs update-service --cluster weather-cluster --service weather-api-service --force-new-deployment
