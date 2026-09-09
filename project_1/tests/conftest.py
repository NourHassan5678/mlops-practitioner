import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.predict import DurationPredictor


@pytest.fixture
def sample_features():
    return {"PULocationID": "43", "DOLocationID": "236", "trip_distance": 2.5}


@pytest.fixture(scope="session")
def trained_model():
    # Session-scoped to prevent reloading the pickle file for every test
    return DurationPredictor()


@pytest.fixture
def client():
    # Uses FastAPI's TestClient to mock API requests
    with TestClient(app) as c:
        yield c
