import pickle
import os
from prodml.config import settings


class DurationPredictor:
    def __init__(self):
        model_path = os.path.join(settings.model_dir, settings.model_name)
        with open(model_path, "rb") as f_in:
            self.dv, self.model = pickle.load(f_in)

    def predict_single(self, features: dict) -> float:
        X = self.dv.transform([features])
        return float(self.model.predict(X)[0])
