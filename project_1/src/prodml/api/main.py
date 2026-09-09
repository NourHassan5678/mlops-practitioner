import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.schemas import BatchTripInput, BatchTripResponse, TripInput

# Keep your custom structured logging imports
from prodml.logging_conf import correlation_id_var, get_logger
from prodml.predict import DurationPredictor

logger = get_logger("prodml.api")
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model at startup rather than on the first request."""
    global predictor
    logger.info("Starting up: Loading model artifact into memory...")
    try:
        predictor = DurationPredictor()
    except Exception as e:  # noqa: BLE001
        logger.error("Startup model load failure", extra={"extra": {"error": str(e)}})
    yield
    logger.info("Shutting down: Releasing model resources...")
    predictor = None


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Maintains your ContextVar-based correlation ID tracking."""
    req_id = str(uuid.uuid4())
    token = correlation_id_var.set(req_id)

    logger.info(f"Incoming request: {request.method} {request.url.path}")

    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    correlation_id_var.reset(token)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("Validation rejection", extra={"extra": {"errors": exc.errors()}})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Validates that the model successfully loaded during startup."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded in memory.",
        )
    return {"status": "healthy", "model_loaded": True}


@app.get("/metadata")
async def get_metadata():
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded in memory.",
        )
    return predictor.get_metadata()


@app.post("/predict")
def predict(trip: TripInput):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model unavailable")

    if trip.trip_distance > 100:
        logger.warning(
            "trip_distance > 100", extra={"extra": {"distance": trip.trip_distance}}
        )

    # Maintain your explicit feature engineering
    features = {
        "PU_DO": f"{trip.PULocationID}_{trip.DOLocationID}",
        "trip_distance": trip.trip_distance,
    }

    logger.debug("feature vector", extra={"extra": {"features": features}})

    start = time.time()
    pred = predictor.predict_single(features)
    latency = time.time() - start

    logger.info(
        "prediction served with latency",
        extra={"extra": {"latency_sec": latency, "prediction": pred}},
    )
    return {"predicted_duration": pred}


@app.post("/predict/batch", response_model=BatchTripResponse)
def predict_batch(payload: BatchTripInput):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model unavailable")

    predictions = []
    start = time.time()

    for trip in payload.inputs:
        # Replicate the feature engineering from your single prediction endpoint
        features = {
            "PU_DO": f"{trip.PULocationID}_{trip.DOLocationID}",
            "trip_distance": trip.trip_distance,
        }

        # Predict and append to the results list
        pred = predictor.predict_single(features)
        predictions.append(pred)

    latency = time.time() - start

    logger.info(
        "batch prediction served",
        extra={
            "extra": {
                "latency_sec": latency,
                "batch_size": len(payload.inputs),
            }
        },
    )

    return {"predicted_durations": predictions}
