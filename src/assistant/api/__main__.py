"""Run the whole platform:  python -m assistant.api"""
import uvicorn

from assistant.api.main import create_app
from assistant.core.config import settings
from assistant.core.errors import ConfigError
from assistant.core.logging import setup_logging


def main() -> None:
    setup_logging(settings.log_level)
    try:
        app = create_app(settings)
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    # log_config=None and access_log=False: our own JSON logs replace uvicorn's.
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
