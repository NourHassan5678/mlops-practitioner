# Module 1 Report & Maturity Self-Assessment

## Performance & Optimization Summary

### 1. Model MAE Metrics
* **Baseline Linear Regression MAE**: 5.12 minutes
* **Final Model MAE**: 5.12 minutes

### 2. Serialization & Latency Comparison
* **Pickle Deserialization + Predict Latency**: 1.84 ms (Mean) | 3.12 ms (P95)
* **ONNX Runtime Execution Latency**: 0.42 ms (Mean) | 0.81 ms (P95)
* **Latency Speedup**: ~4.38x speedup using ONNX Runtime CPU execution.

### 3. Docker Image Footprint
* **Single-Stage Image Size**: 3.36 GB
* **Multi-Stage Image Size**: 906 MB (202 MB compressed push size)
* **Reduction**: 73% reduction in container size.

---

## Serialization Format Matrix

| Format | Human-Readable | Cross-Language | Schema-Enforced | Safe to Load from Untrusted Source |
| :--- | :--- | :--- | :--- | :--- |
| **JSON** | Yes | Yes | No | Yes |
| **Protobuf** | No | Yes | Yes | Yes |
| **Pickle** | No | No (Python only) | No | **No (RCE Vulnerability)** |
| **ONNX** | No | Yes | Yes | Yes |

---

## MLOps Maturity Self-Assessment

### Current Status: Level 1 (DevOps with Manual MLOps)
The project currently operates at Level 1 on the MLOps maturity model. While code packaging, multi-stage containerization, API serving via FastAPI, structured logging, and unit test suites are fully operational, model training, evaluation, and container build steps are still triggered manually.

### Path to Next Level
To reach Level 2 (Automated Pipeline), the system requires automated CI/CD/CT pipelines that trigger continuous integration, automated model retraining upon data ingestion, and experiment tracking with a dedicated model registry. These components will be implemented in Module 2.