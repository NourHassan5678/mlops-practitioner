# prodml: Production-Grade NYC Taxi Duration Prediction Service

`prodml` is an end-to-end machine learning service designed to predict NYC taxi trip durations. Built with Python, scikit-learn, FastAPI, and ONNX Runtime, the project transitions a standalone training workflow into a container-ready, production-grade microservice equipped with structured JSON logging, correlation ID tracing, unit test coverage, and optimized serialization.

---

## Quickstart (Zero to Prediction in 3 Commands)

Execute these three commands in your terminal to start the production container from Docker Hub and serve live predictions:

```bash
# 1. Run the production container from Docker Hub
docker run -d -p 8000:8000 --name prodml-api nourhassan8357/prodml-api:latest

# 2. Check service health status
curl -s http://localhost:8000/health

# 3. Send an inference request
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"PULocationID": 43, "DOLocationID": "236", "trip_distance": "2.5"}'
```

---

## Example API Request & Response

### Request (`POST /predict`)

```json
{
  "PULocationID": 43,
  "DOLocationID": "236",
  "trip_distance": "2.5"
}
```

### Response (`200 OK`)

```json
{
  "prediction_minutes": 11.42,
  "correlation_id": "a3f12b89-8d4e-4e2b-9a0d-2f841c1e9e01",
  "status": "success"
}
```

---

## Repository Structure

```
mlops-practitioner/
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── green_tripdata_2019-08.csv.gz
├── serialization_report.md
├── server_logs.json
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   ├── model.onnx
│   └── baseline.pkl
├── notebooks/
│   └── baseline.ipynb
├── reports/
│   └── module-1.md
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── main.py
│       ├── data/
│       ├── features/
│       └── models/
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_api.py
    ├── test_features.py
    ├── test_predict.py
    └── test_serialization.py
```

---

## 01 — Establish the Baseline

- **Data Processing Pipeline**: Ingests NYC Green Taxi trip datasets, cleans out-of-range records, and calculates trip durations in minutes.
- **Feature Extraction**: Encodes pickup/drop-off location pairs (`PU_DO`) using scikit-learn's `DictVectorizer` alongside `trip_distance`.
- **Baseline Modeling**: Fits a baseline `LinearRegression` model to establish duration predictions and evaluates accuracy against validation metrics.

## 02 — Turn It Into a Real Python Package

- **Package Architecture**: Structured using standard `src/` layout (`src/prodml/`) configured via `pyproject.toml` with strict dependency pinning and optional development toolchains (`pytest`, `ruff`, `black`).
- **CLI Command**: Exposes entrypoint `prodml-train` to run data preprocessing, model fitting, vectorizer persistence, and ONNX export.
- **REST API**: Serves predictions over FastAPI (`POST /predict`) with Pydantic request body validation (`PULocationID`, `DOLocationID`, `trip_distance`).
- **Test Suite & Tooling**: Configured with `pytest`, `pytest-cov`, `black` formatting, and `ruff` linting rules.

## 03 — Structured Logging

- **JSON Formatting**: Custom `JSONFormatter` (`logging.Formatter`) outputs log streams strictly as key-value JSON objects containing `timestamp`, `level`, `logger`, `message`, and `correlation_id`.
- **Correlation ID Tracing**: FastAPI HTTP middleware generates a `uuid4` correlation ID per incoming request, binds it to `contextvars.ContextVar`, and returns it via the `X-Request-ID` HTTP response header.
- **Strict Log Levels**:
  - `DEBUG`: Feature vector details during inference.
  - `INFO`: Successful predictions served with latency measurements.
  - `WARNING`: Inputs exceeding expected boundaries (`trip_distance > 100`).
  - `ERROR`: Model loading failures and request validation rejections.
- **Clean Codebase**: Removed all standard `print()` statements across `src/` in favor of contextual JSON logging.

## 04 — Serialization: Choose Your Format Deliberately

- **ONNX Export**: Converts scikit-learn models and DictVectorizers to ONNX (`skl2onnx`) with dynamic batching on input axes.
- **Parity Verification**: Includes automated tests asserting numerical parity between `.pkl` and `.onnx` outputs across 500 validation rows (`np.allclose(pred_pkl, pred_onnx, atol=1e-4)`).
- **Latency Benchmarking**: Benchmarks mean and p95 inference latencies comparing standard Pickle deserialization against ONNX Runtime CPU execution.
- **Production Format Choice**: Serves production inference requests using ONNX via `onnxruntime` because it eliminates severe remote code execution vulnerabilities of Python Pickle files while offering cross-language support and hardware-optimized inference speeds.

## 05 — Automated Testing & Code Quality

- **Unit & API Testing**: Automated test suite implemented with `pytest` covering API endpoint contracts, Pydantic request body validations, and health check endpoints.
- **Feature & Parity Verification**: Validates DictVectorizer feature mapping logic and asserts output parity between original models and exported ONNX runtimes.
- **Code Formatting & Quality**: Enforces `black` code formatting and `ruff` static code analysis using pre-commit hook configurations.

## 06 — Containerization & Docker Build Optimization

- **Multi-Stage Build**: Utilizes a `python:3.11-slim` multi-stage build pattern to separate build dependencies from runtime layers, reducing image size from 3.36 GB to 906 MB.
- **Context Exclusion**: Employs a root `.dockerignore` file to exclude virtual environments (`.venv`), Git metadata, test caches, notebooks, and raw data files.
- **Non-Root Security Hardening**: Creates and executes container operations under restricted `appuser` credentials (UID 1000) for enhanced security.

## 07 — Container Registry Publishing & Orchestration

- **Orchestration**: Configures `docker-compose.yml` to manage environment variables, local model volume mounts, restart policies, and health probes.
- **Active Health Check**: Implements automated HTTP polling against `/health` at 30-second intervals to monitor API availability.
- **Registry Distribution**: Pushes versioned (`0.1.0`) and `latest` tags to Docker Hub under `nourhassan8357/prodml-api`.