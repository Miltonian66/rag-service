"""Optional Langfuse tracing for LangGraph runs.

Enabled only when the ``observability`` extra is installed *and* both
``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set (``LANGFUSE_HOST`` for
self-hosted). Otherwise this is a no-op, so tests and local runs need nothing.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def langfuse_callbacks() -> list:
    """LangChain-style callbacks to pass in a LangGraph run config."""
    if not langfuse_enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        log.warning("LANGFUSE_* is set but langfuse is missing: uv sync --extra observability")
        return []
    return [CallbackHandler()]
