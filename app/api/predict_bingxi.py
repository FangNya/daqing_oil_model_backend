from flask import Blueprint

from app.api.common import handle_predict


bp = Blueprint("predict_bingxi", __name__)


@bp.post("/predict_bingxi")
def predict_bingxi():
    return handle_predict("bingxi")
