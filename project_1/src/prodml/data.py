import pandas as pd

from prodml.utils import timed


@timed
def read_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, low_memory=False)
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
    df["duration"] = (
        df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df["PULocationID"] = df["PULocationID"].astype(str)
    df["DOLocationID"] = df["DOLocationID"].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    return df


@timed
def split_data(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("lpep_pickup_datetime").reset_index(drop=True)
    cutoff = int(len(df) * train_ratio)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
