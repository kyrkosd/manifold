"""Tests for backend/services/file_parser.py."""
from __future__ import annotations

from pathlib import Path

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
    def test_csv(self):
        assert _detect_file_type(Path("x.csv")) == FileType.CSV

    def test_tsv(self):
        assert _detect_file_type(Path("x.tsv")) == FileType.TSV

    def test_tab(self):
        assert _detect_file_type(Path("x.tab")) == FileType.TSV

    def test_xlsx(self):
        assert _detect_file_type(Path("x.xlsx")) == FileType.XLSX

    def test_unknown_raises(self):
        with pytest.raises(FileParseError, match="Unsupported"):
            _detect_file_type(Path("x.png"))


class TestParseFile:
    def test_csv_row_count(self, csv_path):
        df = parse_file(csv_path)
        assert len(df) == 20

    def test_tsv_row_count(self, tsv_path):
        df = parse_file(tsv_path)
        assert len(df) == 20

    def test_xlsx_row_count(self, xlsx_path):
        df = parse_file(xlsx_path)
        assert len(df) == 20

    def test_semicolon_delimiter(self):
        df = parse_file(FIXTURES / "test_semicolon.csv")
        # Should detect semicolon and still get 7 columns.
        assert len(df.columns) == 7

    def test_no_header(self):
        # No-header CSV gets generated column names (col_0, col_1, ...).
        df = parse_file(FIXTURES / "test_no_header.csv")
        assert all(c.startswith("col_") for c in df.columns)

    def test_encoding_latin1(self):
        df = parse_file(FIXTURES / "test_encodings.csv")
        # Should parse without error; accented values should be present.
        assert len(df) == 3
        assert "name" in df.columns

    def test_invalid_binary_file_raises(self, tmp_path):
        bad = tmp_path / "binary.csv"
        bad.write_bytes(bytes(range(256)) * 100)
        with pytest.raises(FileParseError):
            parse_file(bad)


class TestCleanColumnNames:
    def test_whitespace_stripped(self):
        import pandas as pd
        df = pd.DataFrame({"  foo  ": [1], "bar ": [2]})
        df = _clean_column_names(df)
        assert list(df.columns) == ["foo", "bar"]

    def test_spaces_to_underscores(self):
        import pandas as pd
        df = pd.DataFrame({"hello world": [1]})
        df = _clean_column_names(df)
        assert df.columns[0] == "hello_world"

    def test_duplicates_renamed(self):
        import pandas as pd
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "a"])
        df = _clean_column_names(df)
        assert len(set(df.columns)) == 3
