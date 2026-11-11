import io

import pytest
from docx import Document
from reportlab.pdfgen.canvas import Canvas

from rag.documents import MAX_FILE_BYTES, chunk_document, parse_document


def test_chunks_preserve_offsets_overlap_and_identity():
    text = "abcdefghijklmnopqrstuvwxyz" * 8
    chunks = chunk_document("../notes.txt", text.encode(), chunk_size=50, overlap=10)
    assert chunks[0].source == "notes.txt"
    assert chunks[0].text[-10:] == chunks[1].text[:10]
    assert all(c.text == text[c.start : c.start + 50] for c in chunks)
    assert chunks[-1].start + len(chunks[-1].text) == len(text)
    assert chunks == chunk_document("notes.txt", text.encode(), chunk_size=50, overlap=10)
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[0].id != chunk_document("other.txt", text.encode())[0].id


def test_pdf_keeps_page_provenance():
    buffer = io.BytesIO()
    canvas = Canvas(buffer)
    canvas.drawString(72, 720, "Atlas launch: March 2027")
    canvas.showPage()
    canvas.drawString(72, 720, "Owner: Maya Chen")
    canvas.save()
    chunks = chunk_document("atlas.pdf", buffer.getvalue())
    assert [c.location for c in chunks] == ["page 1", "page 2"]
    assert "Maya Chen" in chunks[1].text


def test_docx_keeps_paragraphs_and_tables_in_order():
    doc = Document()
    doc.add_paragraph("Before table")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Owner"
    table.cell(0, 1).text = "Maya Chen"
    doc.add_paragraph("After table")
    buffer = io.BytesIO()
    doc.save(buffer)
    sections = parse_document("atlas.docx", buffer.getvalue())
    assert [s.location for s in sections] == ["section 1", "section 2", "section 3"]
    assert "Maya Chen" in sections[1].text
    assert sections[2].text == "After table"


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("empty.txt", b"", "empty"),
        ("blank.txt", b"   ", "No readable text"),
        ("wrong.txt", b"\xff", "UTF-8"),
        ("bad.exe", b"binary", "Supported"),
        ("broken.pdf", b"invalid", "Could not read"),
        ("broken.docx", b"invalid", "Could not read"),
        ("big.txt", b"a" * (MAX_FILE_BYTES + 1), "20 MB"),
    ],
)
def test_invalid_documents_are_rejected(name, data, message):
    with pytest.raises(ValueError, match=message):
        parse_document(name, data)


def test_invalid_overlap_and_empty_scanned_pdf():
    with pytest.raises(ValueError, match="overlap"):
        chunk_document("x.txt", b"x", chunk_size=5, overlap=5)
    buffer = io.BytesIO()
    canvas = Canvas(buffer)
    canvas.showPage()
    canvas.save()
    with pytest.raises(ValueError, match="OCR"):
        parse_document("scan.pdf", buffer.getvalue())


def test_utf8_bom_is_removed():
    assert parse_document("notes.txt", b"\xef\xbb\xbfhello")[0].text == "hello"


def test_docx_heading_and_fact_stay_together_with_competing_sections(library):
    doc = Document()
    doc.add_heading("Atlas launch", level=1)
    doc.add_paragraph("15 March 2027")
    for i in range(12):
        doc.add_heading(f"Other project {i}", level=1)
        doc.add_paragraph("Unrelated details " * 25)
    buffer = io.BytesIO()
    doc.save(buffer)
    chunks = chunk_document("projects.docx", buffer.getvalue())
    assert "Atlas launch" in chunks[0].text and "15 March 2027" in chunks[0].text
    assert "through" in chunks[0].location
    library.ingest("projects.docx", buffer.getvalue())
    assert "15 March 2027" in library.search("Atlas launch", k=1)[0].chunk.text


def test_header_only_docx_and_shared_headers_are_indexed_once():
    doc = Document()
    doc.sections[0].header.paragraphs[0].text = "Atlas owner Maya Chen"
    doc.sections[0].footer.paragraphs[0].text = "Retention 90 days"
    doc.add_section()
    buffer = io.BytesIO()
    doc.save(buffer)
    chunks = chunk_document("header-only.docx", buffer.getvalue())
    assert sum("Atlas owner" in c.text for c in chunks) == 1
    assert sum("Retention" in c.text for c in chunks) == 1
    assert {c.location for c in chunks} == {"header 1", "footer 1"}


def test_body_first_page_header_and_footer_tables_have_provenance():
    doc = Document()
    doc.add_paragraph("Body text")
    section = doc.sections[0]
    section.different_first_page_header_footer = True
    section.first_page_header.paragraphs[0].text = "Special cover fact"
    table = section.footer.add_table(rows=1, cols=1, width=1)
    table.cell(0, 0).text = "Footer table fact"
    buffer = io.BytesIO()
    doc.save(buffer)
    chunks = chunk_document("template.docx", buffer.getvalue())
    assert any(c.location == "section 1" and "Body text" in c.text for c in chunks)
    assert any(c.location == "first-page header 1" and "cover fact" in c.text for c in chunks)
    assert any(c.location == "footer 1" and "Footer table fact" in c.text for c in chunks)
