import os

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import settings
from prodml.logging_conf import get_logger

logger = get_logger("prodml.export")


def export_onnx(model, num_features):
    initial_type = [("float_input", FloatTensorType([None, num_features]))]
    onx = convert_sklearn(model, initial_types=initial_type)

    out_path = os.path.join(settings.model_dir, "model.onnx")
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())
    logger.info("Exported ONNX model", extra={"extra": {"path": out_path}})
