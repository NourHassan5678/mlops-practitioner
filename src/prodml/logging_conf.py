import json
import logging
from contextvars import ContextVar

correlation_id_var = ContextVar("correlation_id", default="-")


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_obj.update(record.extra)
        return json.dumps(log_obj)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger
