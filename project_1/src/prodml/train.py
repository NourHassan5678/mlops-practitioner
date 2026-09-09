import os
import pickle

from sklearn.linear_model import LinearRegression

from prodml.config import settings
from prodml.data import read_dataframe, split_data
from prodml.export import export_onnx
from prodml.features import extract_features
from prodml.logging_conf import get_logger

logger = get_logger("prodml.train")


def run_training():
    logger.info("Starting training")
    df = read_dataframe(settings.data_path)
    df_train, df_val = split_data(df)

    X_train, y_train, _X_val, _y_val, dv = extract_features(df_train, df_val)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    os.makedirs(settings.model_dir, exist_ok=True)
    with open(os.path.join(settings.model_dir, settings.model_name), "wb") as f:
        pickle.dump((dv, lr), f)

    num_features = len(dv.get_feature_names_out())
    export_onnx(lr, num_features)

    logger.info("Training complete")


if __name__ == "__main__":
    run_training()
