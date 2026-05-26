"""Tests for backend/services/file_parser.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import pytest

from backend.services.file_parser import (
    FileParseError,
    _clean_column_names,
    _detect_file_type,
    parse_file,
)
from backend.models.enums import FileType

FIXTURES = Path(__file__).parent / "fixtures"


class TestDetectFileType:
    """Verify file-type detection by extension."""

    def test_csv(self):
        """Verify .csv extension maps to FileType.CSV."""
        assert _detect_file_type(Path("x.csv")) == FileType.CSV

    def test_tsv(self):
        """Verify .tsv extension maps to FileType.TSV."""
        assert _detect_file_type(Path("x.tsv")) == FileType.TSV

    def test_tab(self):
        """Verify .tab extension maps to FileType.TSV."""
        assert _detect_file_type(Path("x.tab")) == FileType.TSV

    def test_xlsx(self):
        """Verify .xlsx extension maps to FileType.XLSX."""
        assert _detect_file_type(Path("x.xlsx")) == FileType.XLSX

    def test_unknown_raises(self):
        """Verify an unsupported extension raises FileParseError."""
        with pytest.raises(FileParseError, match="Unsupported"):
            _detect_file_type(Path("x.png"))


class TestParseFile:
    """Verify parsing of CSV, TSV, XLSX, and edge-case files."""

    def test_csv_row_count(self, csv_path):
        """Verify the CSV fixture parses to 20 rows."""
        df = parse_file(csv_path)
        assert len(df) == 20

    def test_tsv_row_count(self, tsv_path):
        """Verify the TSV fixture parses to 20 rows."""
        df = parse_file(tsv_path)
        assert len(df) == 20

    def test_xlsx_row_count(self, xlsx_path):
        """Verify the XLSX fixture parses to 20 rows."""
        df = parse_file(xlsx_path)
        assert len(df) == 20

    def test_semicolon_delimiter(self):
        """Verify semicolon-delimited CSV is auto-detected and gives 7 columns."""
        df = parse_file(FIXTURES / "test_semicolon.csv")
        # Should detect semicolon and still get 7 columns.
        assert len(df.columns) == 7

    def test_no_header(self):
        """Verify header-less CSV gets auto-generated col_N column names."""
        # No-header CSV gets generated column names (col_0, col_1, ...).
        df = parse_file(FIXTURES / "test_no_header.csv")
        assert all(c.startswith("col_") for c in df.columns)

    def test_encoding_latin1(self):
        """Verify latin-1 encoded CSV parses without error."""
        df = parse_file(FIXTURES / "test_encodings.csv")
        # Should parse without error; accented values should be present.
        assert len(df) == 3
        assert "name" in df.columns

    def test_invalid_binary_file_raises(self, tmp_path):
        """Verify a binary file raises FileParseError."""
        bad = tmp_path / "binary.csv"
        bad.write_bytes(bytes(range(256)) * 100)
        with pytest.raises(FileParseError):
            parse_file(bad)


class TestCleanColumnNames:
    """Verify column name normalisation."""

    def test_whitespace_stripped(self):
        """Verify leading/trailing whitespace is removed from column names."""
        df = pd.DataFrame({"  foo  ": [1], "bar ": [2]})
        df = _clean_column_names(df)
        assert list(df.columns) == ["foo", "bar"]

    def test_spaces_to_underscores(self):
        """Verify spaces within column names are replaced with underscores."""
        df = pd.DataFrame({"hello world": [1]})
        df = _clean_column_names(df)
        assert df.columns[0] == "hello_world"

    def test_duplicates_renamed(self):
        """Verify duplicate column names are made unique."""
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "a"])
        df = _clean_column_names(df)
        assert len(set(df.columns)) == 3
