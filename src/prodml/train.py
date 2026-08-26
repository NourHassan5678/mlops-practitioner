import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prodml.config import settings
from prodml.data import read_dataframe, split_data
from prodml.features import extract_features
from prodml.utils import timed


@timed
def run_training():
    df = read_dataframe(settings.data_path)
    df_train, df_val = split_data(df)

    X_train, y_train, X_val, y_val, dv = extract_features(df_train, df_val)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    print(f"Validation MAE: {mean_absolute_error(y_val, y_pred):.4f}")

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse:.4f}")

    os.makedirs(settings.model_dir, exist_ok=True)
    model_path = os.path.join(settings.model_dir, settings.model_name)
    with open(model_path, "wb") as f_out:
        pickle.dump((dv, lr), f_out)


if __name__ == "__main__":
    run_training()
