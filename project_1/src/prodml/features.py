import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from prodml.utils import timed


def prepare_features(input_data: dict) -> dict:
    """Transforms raw dictionary inputs into the feature dictionary
    required by the vectorizer.
    """
    pu = input_data.get("PULocationID", "Unknown")
    do = input_data.get("DOLocationID", "Unknown")
    return {
        "PU_DO": f"{pu}_{do}",
        "trip_distance": float(input_data.get("trip_distance", 0.0)),
    }


@timed
def extract_features(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple:
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dv = DictVectorizer()
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    return X_train, y_train, X_val, y_val, dv
