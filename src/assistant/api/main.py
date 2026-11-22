"""FastAPI gateway: auth, rate limiting, SSE chat, document upload, and MCP on one app."""
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from assistant.api import routes_chat, routes_docs
from assistant.api.middleware import GateMiddleware, RequestIdMiddleware
from assistant.api.services import ServicesFactory, build_services
from assistant.core.auth import Authenticator
from assistant.core.config import Settings, settings
from assistant.core.logging import get_logger, request_id_var
from assistant.core.ratelimit import RateLimiter
from assistant.mcp_server.server import build_mcp_server

log = get_logger(__name__)

STATIC = Path(__file__).parent / "static"
UI_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}


def create_app(cfg: Settings = settings, services_factory: ServicesFactory = build_services) -> FastAPI:
    auth = Authenticator(cfg.api_key_list())  # raises ConfigError if no valid key is configured
    limiter = RateLimiter(cfg.rate_limit_per_minute)
    mcp = build_mcp_server(mounted=True)
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            # Apps added with app.mount() don't get their own lifespan run, so run MCP's here.
            await stack.enter_async_context(mcp.session_manager.run())
            app.state.services = await stack.enter_async_context(services_factory(cfg))
            log.info("api_ready")
            yield

    # Interactive docs are off: the app is not meant to advertise its own surface.
    app = FastAPI(title="Assistant platform", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.cfg = cfg

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Details go to the log; the client gets a generic message and an id to quote.
        log.error("unhandled_error", exc_info=exc)
        rid = request_id_var.get()
        return JSONResponse(
            {"detail": "Internal server error", "request_id": rid},
            status_code=500,
            headers={"X-Request-ID": rid},
        )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers=UI_HEADERS)

    @app.get("/app.js")
    async def app_js() -> FileResponse:
        return FileResponse(STATIC / "app.js", media_type="text/javascript", headers=UI_HEADERS)

    app.include_router(routes_chat.router)
    app.include_router(routes_docs.router)
    app.mount("/", mcp_app)  # last, so every route above wins; this catches /mcp

    # The last middleware added is the outermost: the request id and log line wrap the gate.
    app.add_middleware(GateMiddleware, auth=auth, limiter=limiter)
    app.add_middleware(RequestIdMiddleware)
    return app
