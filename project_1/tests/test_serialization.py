import os
import pickle
import time

import numpy as np
import onnxruntime as rt
import pytest

from prodml.config import settings


def test_pickle_onnx_parity_and_benchmark(capsys):
    """Verifies output parity between Pickle and ONNX formats and benchmarks latency."""
    pkl_path = os.path.join(settings.model_dir, settings.model_name)
    onnx_path = os.path.join(settings.model_dir, "model.onnx")

    # 1. Gracefully skip if model artifacts do not exist yet
    if not os.path.exists(pkl_path):
        pytest.skip("Pickle model file not found.")
    if not os.path.exists(onnx_path):
        pytest.skip("ONNX model file not found.")

    # 2. Load Pickle model & vectorizer
    with open(pkl_path, "rb") as f:
        dv, lr_model = pickle.load(f)

    # 3. Build sample inputs directly with vectorizer (Fast unit test execution)
    num_samples = 100
    sample_dicts = [
        {"PU_DO": f"{i % 50}_{i % 50}", "trip_distance": float((i % 10) + 1)}
        for i in range(num_samples)
    ]
    X_val = dv.transform(sample_dicts).toarray().astype(np.float32)

    # 4. Benchmark Pickle
    t0 = time.perf_counter()
    preds_pkl = lr_model.predict(X_val)
    pkl_time = (time.perf_counter() - t0) * 1000

    # 5. Benchmark ONNX
    sess = rt.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name

    t0 = time.perf_counter()
    onnx_out = sess.run(None, {input_name: X_val})[0]
    preds_onnx = np.asarray(onnx_out).ravel()
    onnx_time = (time.perf_counter() - t0) * 1000

    # 6. Parity Assertion
    np.testing.assert_allclose(preds_pkl, preds_onnx, rtol=1e-4, atol=1e-4)

    # 7. Print Benchmark Results
    with capsys.disabled():
        print(f"\n--- Parity & Latency Report ({num_samples} samples) ---")
        print(f"Pickle Latency: {pkl_time:.3f} ms")
        print(f"ONNX Latency:   {onnx_time:.3f} ms")
