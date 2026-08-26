from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: str = "green_tripdata_2019-08.csv.gz"
    model_dir: str = "../models"
    model_name: str = "baseline.pkl"


settings = Settings()
