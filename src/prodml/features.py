import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from prodml.utils import timed


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
