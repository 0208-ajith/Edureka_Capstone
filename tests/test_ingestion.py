import io

import pandas as pd
import pytest
from pypdf import PdfWriter

from docqa.ingestion import IngestionError, load_document


def test_txt_is_loaded_with_source():
    sections = load_document("notes.txt", "Leave policy: 24 days per year.".encode("utf-8"))
    assert len(sections) == 1
    assert "24 days" in sections[0].text
    assert sections[0].metadata["source"] == "notes.txt"


def test_txt_with_windows_encoding():
    sections = load_document("legacy.txt", "Café menu – €5".encode("cp1252"))
    assert "Café" in sections[0].text


def test_csv_rows_and_summary():
    csv = "Region,Revenue\nNorth,100\nSouth,250\n"
    sections = load_document("sales.csv", csv.encode())
    summary, rows = sections[0], sections[1]
    assert summary.metadata["rows"] == "summary"
    assert "total 350" in summary.text
    assert "Region values: North, South" in summary.text
    assert "Row 2: Region: North; Revenue: 100" in rows.text
    assert rows.metadata["rows"] == "2-3"


def test_semicolon_csv_is_sniffed():
    sections = load_document("eu.csv", "Name;Score\nAna;9\nBo;7\n".encode())
    assert "Name: Ana; Score: 9" in sections[1].text


def test_excel_reads_every_sheet():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"Item": ["Laptop"], "Qty": [3]}).to_excel(writer, sheet_name="Stock", index=False)
        pd.DataFrame({"Name": ["Asha"], "Dept": ["HR"]}).to_excel(writer, sheet_name="Staff", index=False)
    sections = load_document("book.xlsx", buffer.getvalue())
    sheets = {s.metadata["sheet"] for s in sections}
    assert sheets == {"Stock", "Staff"}
    assert any("Item: Laptop; Qty: 3" in s.text for s in sections)


def test_pdf_without_text_is_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(IngestionError, match="No extractable text"):
        load_document("scan.pdf", buffer.getvalue())


def test_corrupt_pdf_is_reported():
    with pytest.raises(IngestionError, match="Could not read"):
        load_document("broken.pdf", b"not really a pdf")


def test_unsupported_and_empty_files():
    with pytest.raises(IngestionError):
        load_document("image.png", b"data")
    with pytest.raises(IngestionError):
        load_document("empty.txt", b"")
