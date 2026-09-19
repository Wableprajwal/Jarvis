PY ?= backend/.venv/bin/python
.PHONY: setup dev test eval ingest up down frontend

setup:
	python3.12 -m venv backend/.venv
	backend/.venv/bin/pip install -r backend/requirements.txt

up:
	docker compose up -d postgres

down:
	docker compose down

dev:
	cd backend && ../$(PY) -m uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm install && npm run dev

test:
	cd backend && ../$(PY) -m pytest -q

eval:
	cd backend && ../$(PY) -m app.eval.ci

ingest:
	cd backend && ../$(PY) -m app.rag.ingest ../docs
