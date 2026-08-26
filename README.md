# MLOps Practitioner - Taxi Duration Prediction

## Development Workflow

### 1. Install
```bash
pip install -e ".[dev]"
ruff check src tests && black --check src tests
pytest -v --cov=src/prodml --cov-report=term-missing
python -m prodml.train
uvicorn prodml.api.main:app --reload --port 8000
