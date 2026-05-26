"""Status and type enumerations for the FMAS import interface."""
from __future__ import annotations

from enum import Enum


class FileType(str, Enum):
    CSV = "csv"
    TSV = "tsv"
    XLSX = "xlsx"


class DataStatus(str, Enum):
    PENDING = "pending"
    PARSED = "parsed"
    PROFILED = "profiled"
    READY = "ready"
    ERROR = "error"


class SuitabilityLevel(str, Enum):
    READY = "ready"
    NEEDS_ATTENTION = "needs_attention"
    NOT_SUITABLE = "not_suitable"


class DatabaseType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
