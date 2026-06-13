from flask import jsonify, request

from app.services.errors import error_response
from app.services.prediction_service import predict_target


def handle_predict(target: str):
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify(predict_target(target, payload))
    except Exception as exc:
        body, status_code = error_response(exc)
        return jsonify(body), status_code
