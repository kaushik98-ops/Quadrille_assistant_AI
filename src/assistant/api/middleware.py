"""Pure ASGI middleware. (Starlette's BaseHTTPMiddleware interferes with streaming responses.)"""
import math
import time

from starlette.responses import JSONResponse

from assistant.core.auth import Authenticator, credential_from_headers
from assistant.core.logging import get_logger, new_request_id
from assistant.core.ratelimit import RateLimiter

log = get_logger(__name__)

# Reachable without a key: the health probe and the static chat page (which holds no secrets).
PUBLIC = {("GET", "/health"), ("GET", "/"), ("GET", "/app.js")}


class RequestIdMiddleware:
    """Gives every request an id (in logs and in the X-Request-ID header) and logs one line."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = new_request_id()
        started = time.monotonic()
        status = 0

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            log.info(
                "request",
                extra={"fields": {
                    "method": scope["method"], "path": scope["path"], "status": status,
                    "ms": round((time.monotonic() - started) * 1000),
                }},
            )


class GateMiddleware:
    """Checks the API key and the rate limit before anything else sees the request.

    Runs ahead of body parsing, so unauthenticated clients can't make the app read uploads.
    """

    def __init__(self, app, auth: Authenticator, limiter: RateLimiter):
        self.app = app
        self._auth = auth
        self._limiter = limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return
        if (scope["method"], scope["path"]) in PUBLIC:
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        key_id = self._auth.identify(
            credential_from_headers(headers.get("authorization"), headers.get("x-api-key"))
        )
        if key_id is None:
            response = JSONResponse(
                {"detail": "Invalid or missing API key"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        wait = self._limiter.check(key_id)
        if wait:
            response = JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(math.ceil(wait))},
            )
            await response(scope, receive, send)
            return

        scope.setdefault("state", {})["key_id"] = key_id  # read back as request.state.key_id
        await self.app(scope, receive, send)
