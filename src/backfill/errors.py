from __future__ import annotations


class AppError(Exception):
    """Base for every error a client may see.

    ``message`` is written for the client and must never contain personal data or internals;
    the error ``code`` is stable and safe to branch on.
    """

    status: int = 500
    code: str = "INTERNAL_ERROR"
    message: str = "Something went wrong. Quote the request id when reporting this."

    def __init__(self, message: str | None = None, *, code: str | None = None) -> None:
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code
        super().__init__(self.message)


class InvalidInput(AppError):
    status = 422
    code = "INVALID_INPUT"
    message = "The request is not valid."


class Unauthorized(AppError):
    status = 401
    code = "UNAUTHORIZED"
    message = "Missing or invalid credentials."


class Forbidden(AppError):
    status = 403
    code = "FORBIDDEN"
    message = "Your role is not allowed to do this."


class NotFound(AppError):
    status = 404
    code = "NOT_FOUND"
    message = "Not found."


class Conflict(AppError):
    status = 409
    code = "CONFLICT"
    message = "The request conflicts with the current state."
