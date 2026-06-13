from flask import Blueprint

from app.api.common import handle_predict


bp = Blueprint("predict_qiya", __name__)


@bp.post("/predict_qiya")
def predict_qiya():
    return handle_predict("qiya")
