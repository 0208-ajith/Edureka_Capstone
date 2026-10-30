"""Load PDF, TXT, CSV and Excel files into text sections and index them (Task 3)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from .chunking import chunk_sections
from .config import SUPPORTED_EXTENSIONS, Settings
from .guardrails import validate_upload

ROWS_PER_SECTION = 25
MAX_DISTINCT_VALUES_LISTED = 20


class IngestionError(ValueError):
    """Raised when a file can't be turned into text. The message is safe to show."""


@dataclass
class Section:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestReport:
    source: str
    sections: int
    chunks: int


def load_document(filename: str, data: bytes) -> list[Section]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"{filename}: unsupported file type '{ext}'.")
    if not data:
        raise IngestionError(f"{filename} is empty.")

    loaders = {".pdf": _load_pdf, ".txt": _load_txt, ".csv": _load_csv, ".xlsx": _load_excel}
    try:
        sections = loaders[ext](data)
    except IngestionError:
        raise
    except Exception as exc:  # parser libraries raise many different exception types
        raise IngestionError(f"Could not read {filename}: {exc}") from exc

    sections = [s for s in sections if s.text.strip()]
    if not sections:
        hint = " Scanned (image-only) PDFs need OCR, which isn't supported." if ext == ".pdf" else ""
        raise IngestionError(f"No extractable text found in {filename}.{hint}")
    for section in sections:
        section.metadata["source"] = filename
    return sections


def ingest_document(filename: str, data: bytes, store, settings: Settings) -> IngestReport:
    """Validate, parse, chunk and index one file, replacing any earlier copy of it."""
    validate_upload(filename, len(data), settings.max_file_mb)
    sections = load_document(filename, data)
    chunks = chunk_sections(sections, settings.chunk_size, settings.chunk_overlap)
    store.replace_source(filename, chunks)
    return IngestReport(source=filename, sections=len(sections), chunks=len(chunks))


def _load_pdf(data: bytes) -> list[Section]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise IngestionError("This PDF is password-protected.")
    return [
        Section(page.extract_text() or "", {"page": number})
        for number, page in enumerate(reader.pages, start=1)
    ]


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _load_txt(data: bytes) -> list[Section]:
    return [Section(_decode_text(data))]


def _load_csv(data: bytes) -> list[Section]:
    # sep=None lets pandas sniff comma / semicolon / tab delimiters.
    df = pd.read_csv(io.StringIO(_decode_text(data)), sep=None, engine="python")
    return _table_sections(df, sheet=None)


def _load_excel(data: bytes) -> list[Section]:
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    sections: list[Section] = []
    for name, df in sheets.items():
        sections.extend(_table_sections(df, sheet=str(name)))
    return sections


def _table_sections(df: pd.DataFrame, sheet: str | None) -> list[Section]:
    """Render a table as a summary section plus row-block sections.

    Rows are written as "column: value" pairs so each chunk is self-describing
    for semantic search. Row numbers match the spreadsheet (header is row 1).
    """
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []
    df.columns = [str(c).strip() for c in df.columns]
    base = {"sheet": sheet} if sheet else {}
    label = f"sheet '{sheet}'" if sheet else "table"

    sections = [Section(_table_summary(df, label), {**base, "rows": "summary"})]
    for start in range(0, len(df), ROWS_PER_SECTION):
        block = df.iloc[start : start + ROWS_PER_SECTION]
        lines = []
        for index, row in block.iterrows():
            cells = "; ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            lines.append(f"Row {int(index) + 2}: {cells}")
        first, last = int(block.index[0]) + 2, int(block.index[-1]) + 2
        sections.append(Section("\n".join(lines), {**base, "rows": f"{first}-{last}"}))
    return sections


def _table_summary(df: pd.DataFrame, label: str) -> str:
    lines = [
        f"Summary of {label}: {len(df)} data rows, {len(df.columns)} columns.",
        "Columns: " + ", ".join(df.columns),
    ]
    for col in df.select_dtypes("number").columns:
        values = df[col].dropna()
        if not values.empty:
            lines.append(
                f"{col}: min {float(values.min()):g}, max {float(values.max()):g}, "
                f"mean {float(values.mean()):g}, total {float(values.sum()):g}"
            )
    for col in df.select_dtypes(exclude="number").columns:
        distinct = df[col].dropna().astype(str).unique()
        if 0 < len(distinct) <= MAX_DISTINCT_VALUES_LISTED:
            lines.append(f"{col} values: " + ", ".join(distinct))
    return "\n".join(lines)
