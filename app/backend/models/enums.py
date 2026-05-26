"""Status and type enumerations for the FMAS import interface."""
from __future__ import annotations

from enum import Enum


class FileType(str, Enum):
    """Supported upload file formats."""

    CSV = "csv"
    TSV = "tsv"
    XLSX = "xlsx"


class DataStatus(str, Enum):
    """Lifecycle states of an imported dataset."""

    PENDING = "pending"
    PARSED = "parsed"
    PROFILED = "profiled"
    READY = "ready"
    ERROR = "error"


class SuitabilityLevel(str, Enum):
    """Dataset suitability rating produced by the data profiler."""

    READY = "ready"
    NEEDS_ATTENTION = "needs_attention"
    NOT_SUITABLE = "not_suitable"


class DatabaseType(str, Enum):
    """Supported database back-ends for SQL import."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
