from pydantic import BaseModel, ConfigDict, Field


class TripInput(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float = Field(..., gt=0)


class BatchTripInput(BaseModel):
    inputs: list[TripInput] = Field(..., description="List of trip prediction requests")


class BatchTripResponse(BaseModel):
    predicted_durations: list[float] = Field(
        ..., description="List of predicted durations"
    )


class PredictionRequest(BaseModel):
    PULocationID: int = Field(..., gt=0, description="Pickup Location ID")
    DOLocationID: int = Field(..., gt=0, description="Dropoff Location ID")
    trip_distance: float = Field(
        ..., gt=0, lt=200, description="Trip distance in miles"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "PULocationID": 43,
                "DOLocationID": 236,
                "trip_distance": 2.5,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted trip duration in minutes")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_version: str = Field(..., description="Model version tag")
    correlation_id: str = Field(..., description="Correlation ID for request tracing")


class BatchPredictionRequest(BaseModel):
    inputs: list[PredictionRequest] = Field(
        ..., min_length=1, description="List of single prediction requests"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": [
                    {"PULocationID": 43, "DOLocationID": 236, "trip_distance": 2.5},
                    {"PULocationID": 132, "DOLocationID": 230, "trip_distance": 12.8},
                ]
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[float] = Field(
        ..., description="List of predicted trip durations in minutes"
    )
    latency_ms: float = Field(
        ..., description="Total batch inference latency in milliseconds"
    )
    model_version: str = Field(..., description="Model version tag")
    correlation_id: str = Field(..., description="Correlation ID for request tracing")
