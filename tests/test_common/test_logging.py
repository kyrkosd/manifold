"""Tests for common/logging.py."""
from __future__ import annotations

import logging

from common.logging import get_logger, setup_logging


def test_get_logger_returns_logger_instance() -> None:
    logger = get_logger("fmas.test")
    assert isinstance(logger, logging.Logger)


def test_get_logger_correct_name() -> None:
    logger = get_logger("fmas.mymodule")
    assert logger.name == "fmas.mymodule"


def test_get_logger_same_name_returns_same_object() -> None:
    a = get_logger("fmas.singleton_check")
    b = get_logger("fmas.singleton_check")
    assert a is b


def test_get_logger_different_names_different_objects() -> None:
    a = get_logger("fmas.alpha")
    b = get_logger("fmas.beta")
    assert a is not b


def test_setup_logging_does_not_raise() -> None:
    setup_logging(level=logging.DEBUG)


def test_setup_logging_idempotent() -> None:
    setup_logging(level=logging.INFO)
    setup_logging(level=logging.WARNING)
    # After two calls the fmas root logger should have at most one handler
    root = logging.getLogger("fmas")
    assert len(root.handlers) <= 1


def test_setup_logging_custom_format_does_not_raise() -> None:
    setup_logging(format="%(levelname)s: %(message)s")
