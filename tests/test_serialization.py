import os
import pickle
import time

import numpy as np
import onnxruntime as rt
import pytest

from prodml.config import settings
from prodml.data import read_dataframe, split_data


def test_onnx_parity_and_benchmark(capsys):
    # 1. Load Data & Pickle Model
    df = read_dataframe(settings.data_path)
    _, df_val = split_data(df)

    model_path = os.path.join(settings.model_dir, settings.model_name)
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet")

    with open(model_path, "rb") as f:
        dv, lr_model = pickle.load(f)

    # Prepare 500 rows
    val_dicts = df_val.head(500)[
        ["PULocationID", "DOLocationID", "trip_distance"]
    ].copy()
    val_dicts["PU_DO"] = (
        val_dicts["PULocationID"].astype(str)
        + "_"
        + val_dicts["DOLocationID"].astype(str)
    )
    X_val = (
        dv.transform(val_dicts[["PU_DO", "trip_distance"]].to_dict(orient="records"))
        .toarray()
        .astype(np.float32)
    )

    # 2. Benchmark Pickle
    latencies_pkl = []
    preds_pkl = []
    for i in range(500):
        t0 = time.perf_counter()
        preds_pkl.append(lr_model.predict(X_val[i : i + 1])[0])
        latencies_pkl.append(time.perf_counter() - t0)

    # 3. Benchmark ONNX
    sess = rt.InferenceSession(
        os.path.join(settings.model_dir, "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    latencies_onnx = []
    preds_onnx = []
    for i in range(500):
        t0 = time.perf_counter()
        preds_onnx.append(sess.run(None, {input_name: X_val[i : i + 1]})[0][0][0])
        latencies_onnx.append(time.perf_counter() - t0)

    # 4. Parity Test (Assertion)
    assert np.allclose(preds_pkl, preds_onnx, atol=1e-4)

    # 5. Output Report
    mean_pkl, p95_pkl = (
        np.mean(latencies_pkl) * 1000,
        np.percentile(latencies_pkl, 95) * 1000,
    )
    mean_onnx, p95_onnx = (
        np.mean(latencies_onnx) * 1000,
        np.percentile(latencies_onnx, 95) * 1000,
    )

    with capsys.disabled():
        print("\n--- Benchmark (500 rows) ---")
        print(f"Pickle -> Mean: {mean_pkl:.3f}ms | p95: {p95_pkl:.3f}ms")
        print(f"ONNX   -> Mean: {mean_onnx:.3f}ms | p95: {p95_onnx:.3f}ms")
