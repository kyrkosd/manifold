"""CSV, TSV, and Excel file parsing for the FMAS import interface."""
from __future__ import annotations

import csv
import logging
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import UploadFile

from backend.models.enums import FileType

log = logging.getLogger(__name__)

MAX_ROWS: int = 500_000
_ENCODINGS = ("utf-8", "latin-1", "cp1252")


class FileParseError(Exception):
    """Raised when a file cannot be parsed."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(file_path: Path, file_type: FileType | None = None) -> pd.DataFrame:
    """Parse *file_path* into a DataFrame. Auto-detects type from extension."""
    t0 = time.monotonic()
    size_mb = file_path.stat().st_size / 1_048_576
    ft = file_type or _detect_file_type(file_path)
    log.info("Parsing %s (%.1f MB) as %s.", file_path.name, size_mb, ft.value)

    if ft == FileType.CSV:
        df = _parse_csv(file_path)
    elif ft == FileType.TSV:
        df = _parse_tsv(file_path)
    else:
        df = _parse_excel(file_path)

    df = _clean_column_names(df)
    rows, cols = df.shape
    duration = time.monotonic() - t0

    log.info("Parsed %d rows, %d columns in %.2fs.", rows, cols, duration)

    return df


async def save_upload(upload_file: UploadFile, data_dir: Path) -> Path:
    """Save *upload_file* to *data_dir* with a UUID filename; return the path."""
    suffix = Path(upload_file.filename or "upload").suffix.lower()
    dest = data_dir / f"{uuid.uuid4().hex}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = await upload_file.read()
    dest.write_bytes(content)

    return dest


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------

def _detect_file_type(file_path: Path) -> FileType:
    ext = file_path.suffix.lower()
    if ext == ".csv":
        return FileType.CSV
    if ext in (".tsv", ".tab"):
        return FileType.TSV
    if ext in (".xlsx", ".xls"):
        return FileType.XLSX
    raise FileParseError(f"Unsupported file extension: {ext!r}. Expected .csv, .tsv, .xlsx, or .xls.")


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------

def _parse_csv(file_path: Path, delimiter: str | None = None) -> pd.DataFrame:
    encoding = _detect_encoding(file_path)
    sep = delimiter or _sniff_delimiter(file_path, encoding)
    has_header = _sniff_has_header(file_path, encoding, sep)
    try:
        df = pd.read_csv(
            file_path,
            sep=sep,
            encoding=encoding,
            nrows=MAX_ROWS,
            on_bad_lines="skip",
            header=0 if has_header else None,
        )
    except Exception as exc:
        raise FileParseError(f"Could not parse CSV: {exc}") from exc
    if df.empty and df.columns.tolist() == ["Unnamed: 0"]:
        raise FileParseError("File does not appear to contain tabular data.")
    if not has_header:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df


def _parse_tsv(file_path: Path) -> pd.DataFrame:
    return _parse_csv(file_path, delimiter="\t")


def _parse_excel(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(
            file_path,
            engine="openpyxl",
            nrows=MAX_ROWS,
            sheet_name=0,
        )
    except Exception as exc:
        raise FileParseError(f"Could not parse Excel file: {exc}") from exc
    # Drop leading fully-empty rows that openpyxl sometimes includes.
    first_real = df.apply(lambda r: r.notna().any(), axis=1).idxmax()
    if first_real > 0:
        df = df.iloc[first_real:].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sniff_delimiter(file_path: Path, encoding: str) -> str:
    """Detect CSV delimiter from the first 10 KB using csv.Sniffer."""
    try:
        sample = file_path.read_bytes()[:10_240].decode(encoding, errors="replace")
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _sniff_has_header(file_path: Path, encoding: str, delimiter: str) -> bool:
    """Return True if the file appears to have a header row.

    csv.Sniffer.has_header() can produce false negatives on small files where all
    first-row values are non-numeric identifiers.  The fallback checks whether every
    cell in the first row looks like a text label rather than a number.
    """
    try:
        sample = file_path.read_bytes()[:10_240].decode(encoding, errors="replace")
        if csv.Sniffer().has_header(sample):
            return True
        lines = [ln for ln in sample.splitlines() if ln.strip()]
        if not lines:
            return True
        first_row = next(csv.reader([lines[0]], delimiter=delimiter))
        # If every cell in the first row is a non-numeric string, it's a header.
        return all(_looks_like_label(cell) for cell in first_row)
    except csv.Error:
        return True


def _looks_like_label(cell: str) -> bool:
    """True when *cell* cannot be interpreted as a number."""
    try:
        float(cell.strip())
        return False
    except ValueError:
        return bool(cell.strip())


def _detect_encoding(file_path: Path) -> str:
    """Try UTF-8, then latin-1, then cp1252; return the first that decodes cleanly."""
    sample = file_path.read_bytes()[:10_240]
    if b"\x00" in sample:
        raise FileParseError("File appears to be binary (contains null bytes).")
    for enc in _ENCODINGS:
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, lowercase, underscores, deduplicate."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for i, col in enumerate(df.columns):
        col_str = str(col).strip().lower().replace(" ", "_")
        if not col_str or col_str.startswith("unnamed"):
            col_str = f"col_{i}"
        if col_str in seen:
            seen[col_str] += 1
            col_str = f"{col_str}_{seen[col_str]}"
        else:
            seen[col_str] = 0
        names.append(col_str)
    df.columns = names  # type: ignore[assignment]
    return df
