from flask import Blueprint

from app.api.common import handle_predict


bp = Blueprint("predict_C5", __name__)


@bp.post("/predict_C5")
def predict_C5():
    return handle_predict("C5")
