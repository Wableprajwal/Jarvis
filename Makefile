PY ?= backend/.venv/bin/python
.PHONY: setup setup-db dev lint test eval eval-live eval-regression-demo ingest up down frontend frontend-test check clean

setup:
	python3.12 -m venv backend/.venv
	backend/.venv/bin/pip install -r backend/requirements.txt

## Create the separate database the test suite uses, so tests never write to the dev database.
## Only needed when pointing at a Postgres whose role cannot CREATE DATABASE (compose's can).
setup-db:
	@name=$$($(PY) -c "import sys;sys.path.insert(0,'backend');from app.config import get_settings;print(get_settings().database_url.rsplit('/',1)[1]+'_test')"); \
	echo "creating test database $$name (safe to re-run)"; \
	createdb "$$name" 2>/dev/null || echo "  already exists, or create it manually: createdb $$name"

up:
	docker compose up -d postgres

down:
	docker compose down

dev:
	cd backend && ../$(PY) -m uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm install && npm run dev

frontend-test:
	cd frontend && npm test -- --run && npm run build

lint:
	cd backend && ../$(PY) -m ruff check app tests

## Everything CI would run: lint, the whole test suite, and the offline eval gate.
check: lint test eval

## Whole suite. No live model needed: MockProvider covers the core logic and the Ollama-backed
## tests skip themselves when Ollama is not running.
test:
	cd backend && ../$(PY) -m pytest -q

## CI gate: the two-layer eval suite, offline against MockProvider. Non-zero exit on regression.
eval:
	cd backend && ../$(PY) -m app.eval.ci

## Same suite against the real model (needs Ollama); slower, and the number that actually matters.
eval-live:
	cd backend && ../$(PY) -m app.eval.ci --live

## Prove the gate works: deliberately break a prompt and watch the eval fail (expected non-zero).
eval-regression-demo:
	cd backend && ../$(PY) -m app.eval.ci --inject-regression || echo "^ gate correctly failed"

ingest:
	cd backend && ../$(PY) -m app.rag.ingest ../docs

clean:
	find backend -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.pytest_cache frontend/dist
