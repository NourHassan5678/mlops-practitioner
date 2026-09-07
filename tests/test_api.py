import prodml.api.main


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predict_happy_path(client, monkeypatch):
    class MockPredictor:
        def predict_single(self, features):
            return 12.5

    monkeypatch.setattr(prodml.api.main, "predictor", MockPredictor())

    payload = {"PULocationID": "43", "DOLocationID": "236", "trip_distance": 2.5}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"predicted_duration": 12.5}


def test_predict_batch_happy_path(client, monkeypatch):
    class MockPredictor:
        def predict_single(self, features):
            return 10.0

    monkeypatch.setattr(prodml.api.main, "predictor", MockPredictor())

    payload = {
        "inputs": [
            {"PULocationID": "43", "DOLocationID": "236", "trip_distance": 2.5},
            {"PULocationID": "132", "DOLocationID": "230", "trip_distance": 5.0},
        ]
    }
    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    assert response.json() == {"predicted_durations": [10.0, 10.0]}


def test_predict_invalid_payload(client):
    payload = {"PULocationID": "43", "DOLocationID": "236"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
