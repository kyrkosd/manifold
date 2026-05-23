"""
Structured logging setup for FMAS: provides a consistent logger factory
and a one-time root-logger configuration used at application entry points.
"""
from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, creating it if necessary.

    Parameters
    ----------
    name:
        Logger name; pass ``__name__`` from each module for consistent hierarchy.

    Returns
    -------
    logging.Logger
        The named logger instance.
    """
    return logging.getLogger(name)


def setup_logging(
    level: int | str = logging.INFO,
    log_format: str | None = None,
) -> None:
    """Configure the root ``fmas`` logger.

    Idempotent: repeated calls have no effect after the first successful call.

    Parameters
    ----------
    level:
        Logging level (e.g. ``logging.DEBUG`` or the string ``"WARNING"``).
    log_format:
        Log record format string; defaults to the FMAS standard format.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(log_format or _DEFAULT_FORMAT))
    root = logging.getLogger("fmas")
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True
