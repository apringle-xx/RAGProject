"""Parse documents without saving originals; every chunk retains its provenance."""

import hashlib
import io
import zipfile
from pathlib import PurePath

from docx import Document
from pypdf import PdfReader

from rag.types import Chunk, Section

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000
MAX_CHUNKS = 2000


def document_id(filename: str, data: bytes) -> str:
    return hashlib.sha256(filename.encode() + b"\0" + data).hexdigest()


def parse_document(filename: str, data: bytes) -> list[Section]:
    if not data:
        raise ValueError("The document is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Documents must be 20 MB or smaller.")
    extension = PurePath(filename).suffix.lower()
    try:
        if extension == ".txt":
            sections = [Section("text", data.decode("utf-8-sig"))]
        elif extension == ".pdf":
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported. Upload an unlocked copy.")
            if len(reader.pages) > 500:
                raise ValueError("PDFs must have at most 500 pages.")
            sections = [
                Section(f"page {i}", page.extract_text() or "")
                for i, page in enumerate(reader.pages, 1)
            ]
        elif extension == ".docx":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if sum(item.file_size for item in archive.infolist()) > MAX_FILE_BYTES * 5:
                    raise ValueError("The expanded DOCX is too large.")
            doc = Document(io.BytesIO(data))
            # Include table cells in document order, not just paragraphs.
            sections = []
            for i, block in enumerate(doc.iter_inner_content(), 1):
                if hasattr(block, "rows"):
                    text = "\n".join(
                        " | ".join(cell.text for cell in row.cells) for row in block.rows
                    )
                else:
                    text = block.text
                sections.append(Section(f"section {i}", text))
        else:
            raise ValueError("Supported file types are PDF, DOCX and UTF-8 TXT.")
    except UnicodeDecodeError as exc:
        raise ValueError("Text files must use UTF-8 encoding.") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Could not read this document. Check that it is a valid file.") from exc
    if sum(len(s.text) for s in sections) > MAX_TEXT_CHARS:
        raise ValueError("The extracted document is too large (maximum 2 million characters).")
    sections = [Section(s.location, s.text.replace("\x00", "").strip()) for s in sections]
    sections = [s for s in sections if s.text]
    if not sections:
        raise ValueError("No readable text found. Scanned PDFs need OCR before upload.")
    return sections


def chunk_document(
    filename: str, data: bytes, *, chunk_size: int = 1800, overlap: int = 250
) -> list[Chunk]:
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("Chunk size must be positive and overlap smaller than the chunk size.")
    # Normalize names for display only; never use uploaded paths as filesystem paths.
    source = PurePath(filename.replace("\\", "/")).name
    doc_id = document_id(source, data)
    chunks = []
    for section in parse_document(source, data):
        for start in range(0, len(section.text), chunk_size - overlap):
            text = section.text[start : start + chunk_size]
            chunk_id = hashlib.sha256(f"{doc_id}:{section.location}:{start}".encode()).hexdigest()
            chunks.append(Chunk(chunk_id, doc_id, source, section.location, start, text))
            if len(chunks) > MAX_CHUNKS:
                raise ValueError(
                    "The document produces too many chunks; split it into smaller files."
                )
            if start + chunk_size >= len(section.text):
                break
    return chunks
