import os
import pickle

from prodml.config import settings


class DurationPredictor:
    def __init__(self):
        with open(os.path.join(settings.model_dir, settings.model_name), "rb") as f:
            self.dv, self.model = pickle.load(f)

    def predict_single(self, features: dict) -> float:
        X = self.dv.transform([features])
        return float(self.model.predict(X)[0])
