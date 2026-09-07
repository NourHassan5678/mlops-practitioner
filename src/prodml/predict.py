import os
import pickle

from prodml.config import settings


class DurationPredictor:
    def __init__(self):
        print(">>> INITIALIZING DURATION PREDICTOR AND LOADING PICKLE <<<")
        with open(os.path.join(settings.model_dir, settings.model_name), "rb") as f:
            self.dv, self.model = pickle.load(f)

    def predict_single(self, features: dict) -> float:
        X = self.dv.transform([features])
        return float(self.model.predict(X)[0])

    def get_metadata(self):
        """Returns model metadata for the API endpoint."""
        return {
            "model_name": "NYC Taxi Trip Duration Predictor",
            "version": "0.1.0",
            "description": """Predicts taxi trip duration based on pickup location,
dropoff location, and distance.
""",
        }
