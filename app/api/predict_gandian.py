from flask import Blueprint

from app.api.common import handle_predict


bp = Blueprint("predict_gandian", __name__)


@bp.post("/predict_gandian")
def predict_gandian():
    return handle_predict("gandian")
