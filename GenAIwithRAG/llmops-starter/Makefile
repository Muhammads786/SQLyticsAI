SHELL := /bin/bash
.PHONY: up down ingest eval logs

up:
	docker compose up -d --build

down:
	docker compose down -v

ingest:
	docker compose exec api python scripts/ingest.py data

eval:
	docker compose exec api python evals/ragas_eval.py

logs:
	docker compose logs -f api
