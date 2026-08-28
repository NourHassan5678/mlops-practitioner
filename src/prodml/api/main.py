import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from prodml.logging_conf import correlation_id_var, get_logger
from prodml.predict import DurationPredictor

logger = get_logger("prodml.api")
app = FastAPI()
predictor = None


class TripInput(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float = Field(..., gt=0)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
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


def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = DurationPredictor()
        except Exception as e:
            logger.error("Model load failure", extra={"extra": {"error": str(e)}})
            raise HTTPException(status_code=500, detail="Model unavailable") from e
    return predictor


@app.post("/predict")
def predict(trip: TripInput):
    if trip.trip_distance > 100:
        logger.warning(
            "trip_distance > 100", extra={"extra": {"distance": trip.trip_distance}}
        )

    features = {
        "PU_DO": f"{trip.PULocationID}_{trip.DOLocationID}",
        "trip_distance": trip.trip_distance,
    }

    logger.debug("feature vector", extra={"extra": {"features": features}})

    start = time.time()
    pred = get_predictor().predict_single(features)
    latency = time.time() - start

    logger.info(
        "prediction served with latency",
        extra={"extra": {"latency_sec": latency, "prediction": pred}},
    )
    return {"predicted_duration": pred}
