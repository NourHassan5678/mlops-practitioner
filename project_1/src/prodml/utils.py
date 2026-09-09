import time
from functools import wraps

from prodml.logging_conf import get_logger

logger = get_logger("prodml.utils")


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = round(time.time() - start, 4)
        logger.info(
            f"Function {func.__name__} completed",
            extra={"extra": {"function": func.__name__, "duration_sec": duration}},
        )
        return result

    return wrapper
