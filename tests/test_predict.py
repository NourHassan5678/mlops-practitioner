def test_prediction_properties(trained_model, sample_features):
    features = {
        "PU_DO": f"{sample_features['PULocationID']}_{sample_features['DOLocationID']}",
        "trip_distance": sample_features["trip_distance"],
    }

    pred1 = trained_model.predict_single(features)
    pred2 = trained_model.predict_single(features)

    # 1. Returns a float
    assert isinstance(pred1, float)

    # 2. Sane range (e.g., a taxi trip is usually between 1 and 500 minutes)
    assert 0.0 < pred1 < 500.0

    # 3. Deterministic across two calls
    assert pred1 == pred2
