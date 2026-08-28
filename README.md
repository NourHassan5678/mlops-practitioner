# prodml: Production-Grade Taxi Duration Prediction Service

`prodml` is an end-to-end machine learning service designed to predict NYC taxi trip durations. Built with Python, scikit-learn, FastAPI, and ONNX Runtime, the project transitions a standalone training workflow into a container-ready, production-grade microservice equipped with structured JSON logging, correlation ID tracing, unit test coverage, and optimized serialization.

---

## 01 Establish the baseline

* **Data Processing Pipeline**: Ingests NYC Green Taxi trip datasets, cleans out-of-range records, and calculates trip durations in minutes.
* **Feature Extraction**: Encodes pickup/drop-off location pairs (`PU_DO`) using scikit-learn's `DictVectorizer` alongside `trip_distance`.
* **Baseline Modeling**: Fits a baseline `LinearRegression` model to establish duration predictions and evaluates accuracy against validation metrics.

---

## 02 Turn it into a real Python package

* **Package Architecture**: Structured using standard `src/` layout (`src/prodml/`) configured via `pyproject.toml` with strict dependency pinning and optional development toolchains (`pytest`, `ruff`, `black`).
* **CLI Command**: Exposes entrypoint `prodml-train` to run data preprocessing, model fitting, vectorizer persistence, and ONNX export.
* **REST API**: Serves predictions over FastAPI (`POST /predict`) with Pydantic request body validation (`PULocationID`, `DOLocationID`, `trip_distance`).
* **Test Suite & Tooling**: Configured with `pytest`, `pytest-cov`, `black` formatting, and `ruff` linting rules.

---

## 03 Structured logging

* **JSON Formatting**: Custom `JSONFormatter` (`logging.Formatter`) outputs log streams strictly as key-value JSON objects containing `timestamp`, `level`, `logger`, `message`, and `correlation_id`.
* **Correlation ID Tracing**: FastAPI HTTP middleware generates a `uuid4` correlation ID per incoming request, binds it to `contextvars.ContextVar`, and returns it via the `X-Request-ID` HTTP response header.
* **Strict Log Levels**:
  * **`DEBUG`**: Feature vector details during inference.
  * **`INFO`**: Successful predictions served with latency measurements.
  * **`WARNING`**: Inputs exceeding expected boundaries (`trip_distance > 100`).
  * **`ERROR`**: Model loading failures and request validation rejections.
* **Clean Codebase**: Removed all standard `print()` statements across `src/` in favor of contextual JSON logging.

---

## 04 Serialization: choose your format deliberately

* **ONNX Export**: Converts scikit-learn models and DictVectorizers to ONNX (`skl2onnx`) with dynamic batching on input axes.
* **Parity Verification**: Includes automated tests asserting numerical parity between `.pkl` and `.onnx` outputs across 500 validation rows (`np.allclose(pred_pkl, pred_onnx, atol=1e-4)`).
* **Latency Benchmarking**: Benchmarks mean and p95 inference latencies comparing standard Pickle deserialization against ONNX Runtime CPU execution.

### Serialization Format Matrix

| Format | Human-Readable | Cross-Language | Schema-Enforced | Safe to Load from Untrusted Source |
| :--- | :--- | :--- | :--- | :--- |
| **JSON** | Yes | Yes | No | Yes |
| **Protobuf** | No | Yes | Yes | Yes |
| **Pickle** | No | No (Python only) | No | **No (Arbitrary Code Execution)** |
| **ONNX** | No | Yes | Yes | Yes |

> **Security Warning**: **Pickle executes arbitrary code on load. Never load a `.pkl` file you did not produce yourself.**

**Production Format Choice**: This service serves production inference requests using **ONNX** via `onnxruntime` because it eliminates the severe remote code execution vulnerabilities of Python Pickle files while offering cross-language support and hardware-optimized inference speeds.
