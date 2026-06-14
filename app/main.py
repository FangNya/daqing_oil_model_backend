from flask import Flask, jsonify

from app.api.predict_C5 import bp as predict_C5_bp
from app.api.predict_bingxi import bp as predict_bingxi_bp
from app.api.predict_gandian import bp as predict_gandian_bp
from app.api.predict_qiya import bp as predict_qiya_bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(predict_bingxi_bp)
    app.register_blueprint(predict_C5_bp)
    app.register_blueprint(predict_gandian_bp)
    app.register_blueprint(predict_qiya_bp)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
