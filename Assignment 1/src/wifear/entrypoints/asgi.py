"""The entrypoint for the ASGI (FastAPI) application.

This module is used only by an external ASGI application server like `uvicorn` to start the web
service, including any setup like logging etc. that needs to be completed before the service is
ready to accept requests.

    `uvicorn wifear.entrypoints.asgi:app --reload`

"""

from wifear.core.logging import setup_logging
from wifear.entrypoints.api.app import app

__all__ = ["app"]

setup_logging()
