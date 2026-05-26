.PHONY: dev backend frontend test clean

dev:
	@echo "Run 'make backend' and 'make frontend' in separate shells."

backend:
	cd backend && .venv/Scripts/python -m uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && .venv/Scripts/python -m pytest
	cd frontend && npm test

clean:
	rm -rf backend/.venv backend/weights frontend/node_modules frontend/dist
