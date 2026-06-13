from __future__ import annotations


class BackendError(Exception):
    status_code = 400

    def __init__(self, code: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        if status_code is not None:
            self.status_code = status_code


def error_response(error: Exception):
    if isinstance(error, BackendError):
        return {
            "status": "error",
            "code": error.code,
            "message": error.message,
        }, error.status_code
    return {
        "status": "error",
        "code": "INTERNAL_ERROR",
        "message": str(error),
    }, 500
