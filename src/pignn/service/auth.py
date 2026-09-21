"""Service-to-service authentication for the internal FastAPI deployment."""
from __future__ import annotations

import hmac
import os
from typing import Annotated

from fastapi import Header, HTTPException, status


def require_internal_api_key(x_geoargus_internal_key: Annotated[str | None, Header()] = None) -> None:
    """Require the server-side secret sent only by the Next.js backend."""
    expected = os.getenv("PIGNN_INTERNAL_API_KEY")
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="PIGNN_INTERNAL_API_KEY is not configured")
    if x_geoargus_internal_key is None or not hmac.compare_digest(x_geoargus_internal_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid internal service credential")
