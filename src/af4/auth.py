"""
AF4 — API key authentication + rate limiting.

Simple token-based auth: keys stored in an env var (comma-separated)
or in a JSON file at AF4_KEYS_PATH. Rate limit: 10 inference
requests per minute per key.
"""
from __future__ import annotations

import os
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_RATE_WINDOW    = 60       # seconds
_RATE_LIMIT     = 10       # requests per window
_INFERENCE_PATHS = {"/api/v1/infer", "/api/v1/explorer", "/api/v1/cf4", "/api/v1/af1"}

# in-memory rate counter: {key: [(timestamp, count), ...]}
_rate_store: dict[str, list[float]] = defaultdict(list)


def _load_keys() -> set[str]:
    """Load valid API keys from env or JSON file."""
    keys: set[str] = set()

    env_keys = os.environ.get("SHEAR_API_KEYS", "")
    for k in env_keys.split(","):
        k = k.strip()
        if k:
            keys.add(k)

    keys_path = os.environ.get("AF4_KEYS_PATH", "")
    if keys_path and Path(keys_path).exists():
        with open(keys_path) as f:
            data = json.load(f)
            keys.update(data.get("keys", []))

    # if no keys configured, auth is disabled (open access)
    return keys


def _check_rate(api_key: str) -> None:
    now    = time.time()
    window = _rate_store[api_key]
    # prune old entries
    _rate_store[api_key] = [t for t in window if now - t < _RATE_WINDOW]
    if len(_rate_store[api_key]) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {_RATE_LIMIT} requests per {_RATE_WINDOW}s per API key.",
        )
    _rate_store[api_key].append(now)


def get_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> Optional[str]:
    """
    FastAPI dependency. Validates API key if any are configured.
    Returns the key (or None if auth is disabled).
    """
    valid_keys = _load_keys()
    if not valid_keys:
        return None  # auth disabled

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key.")

    _check_rate(api_key)
    return api_key
