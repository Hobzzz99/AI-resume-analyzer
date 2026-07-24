"""Document loading with page attribution.

Extraction is the stage where real-world failures actually happen: encrypted
PDFs, corrupt byte streams, and — most commonly — scans that parse perfectly and
yield nothing. Each of those gets a distinct typed error here (FR-004), because
"analysis failed" is not a message anyone can act on, whereas "this file appears
to be an image-only scan; OCR is not supported" is.

Page attribution is preserved from the first stage onward. It cannot be
reconstructed later, and without it a citation degrades to a filename, which
fails SC-004.

Heavy third-party imports are deliberately function-local. Importing
``langchain_community`` at module scope costs roughly a second and pulls in a
dependency tree that the text-only path never needs — and it would make
``import app.rag.loaders`` in a unit test slower than the test itself.
"""

from __future__ import annotations

from pathlib import Path

from app.schemas.rag import SourceDocument
from app.utils.exceptions import (
    EmptyDocumentError,
    InvalidDocumentError,
    UnsupportedFileTypeError,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PDFLoader:
    """Extracts text from a PDF, one :class:`SourceDocument` per page.

    Backed by LangChain's ``PyPDFLoader`` (pypdf underneath). Chosen over
    ``pdfplumber`` and ``PyMuPDF`` because it is pure-Python with no system
    dependency, and it is the loader the brief names — for resumes, which are
    linear text documents, its extraction quality is indistinguishable from the
    alternatives.
    """

    extensions = frozenset({".pdf"})

    def supports(self, path_or_name: str) -> bool:
        """Whether this loader handles the given filename."""
        return Path(path_or_name).suffix.lower() in self.extensions

    def load(self, path: str) -> list[SourceDocument]:
        """Extract every page of the PDF.

        Raises:
            InvalidDocumentError: The file is corrupt, encrypted, or unreadable.
            EmptyDocumentError: The file parsed but produced no text at all,
                which for a PDF means it is image-only.
        """
        from langchain_community.document_loaders import PyPDFLoader  # noqa: PLC0415

        file_path = Path(path)
        try:
            pages = PyPDFLoader(str(file_path)).load()
        except Exception as exc:
            logger.error(
                "pdf extraction failed",
                extra={"source_file": file_path.name, "error": str(exc)},
            )
            raise InvalidDocumentError(
                f"Could not read '{file_path.name}'. The file may be corrupt, "
                f"password-protected, or not a valid PDF.",
                details={"filename": file_path.name, "reason": str(exc)},
            ) from exc

        documents = [
            SourceDocument(
                text=page.page_content,
                filename=file_path.name,
                # PyPDFLoader pages are 0-based; humans and citations are 1-based.
                page=int(page.metadata.get("page", index)) + 1,
            )
            for index, page in enumerate(pages)
            if page.page_content and page.page_content.strip()
        ]

        if not documents:
            raise EmptyDocumentError(
                f"No extractable text found in '{file_path.name}'. The file appears to be "
                f"image-only; OCR is not supported.",
                details={"filename": file_path.name, "pages": len(pages)},
            )

        logger.info(
            "loaded pdf",
            extra={"stage": "load", "source_file": file_path.name, "pages": len(documents)},
        )
        return documents


class TextLoader:
    """Loads a plain-text or Markdown file as a single unpaginated document.

    ``page`` is 0 rather than 1: the document genuinely has no pages, and
    inventing "page 1" would make a citation claim precision the source does not
    have. Downstream code treats 0 as "unpaginated" (see ``ChunkMetadata.page``).
    """

    extensions = frozenset({".txt", ".md", ".markdown"})

    def supports(self, path_or_name: str) -> bool:
        """Whether this loader handles the given filename."""
        return Path(path_or_name).suffix.lower() in self.extensions

    def load(self, path: str) -> list[SourceDocument]:
        """Read the file, tolerating imperfect encodings.

        Raises:
            InvalidDocumentError: The file could not be read from disk.
            EmptyDocumentError: The file is empty or whitespace only.
        """
        file_path = Path(path)
        try:
            # errors="replace" rather than strict: a single bad byte in an
            # otherwise good resume should not fail the whole upload.
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise InvalidDocumentError(
                f"Could not read '{file_path.name}': {exc}",
                details={"filename": file_path.name},
            ) from exc

        if not text.strip():
            raise EmptyDocumentError(
                f"'{file_path.name}' is empty.", details={"filename": file_path.name}
            )

        logger.info(
            "loaded text file",
            extra={"stage": "load", "source_file": file_path.name, "chars": len(text)},
        )
        return [SourceDocument(text=text, filename=file_path.name, page=0)]


class RawTextLoader:
    """Wraps an in-memory string as a document.

    Exists so that a pasted job description travels the identical ingestion path
    as an uploaded file (FR-001, US2 scenario 5). The alternative — writing the
    paste to a temp file so the file loader can read it back — would be a
    round-trip through the filesystem purely to satisfy an interface.
    """

    def __init__(self, text: str, filename: str = "pasted_text.txt") -> None:
        self._text = text
        self._filename = filename

    def supports(self, path_or_name: str) -> bool:  # noqa: ARG002 - protocol conformance
        """Always true: this loader is constructed for one specific payload."""
        return True

    def load(self, path: str = "") -> list[SourceDocument]:  # noqa: ARG002 - protocol conformance
        """Return the wrapped text as a single document.

        Raises:
            EmptyDocumentError: The supplied text is empty or whitespace only.
        """
        if not self._text.strip():
            raise EmptyDocumentError(
                "The submitted text is empty.", details={"filename": self._filename}
            )
        return [SourceDocument(text=self._text, filename=self._filename, page=0)]


class LoaderRegistry:
    """Dispatches a file to the loader that handles its type.

    New formats (DOCX, HTML) are added by registering a loader here; nothing
    else in the pipeline changes. That is Principle V applied at the extraction
    boundary.
    """

    def __init__(self, loaders: list[PDFLoader | TextLoader] | None = None) -> None:
        self._loaders: list[PDFLoader | TextLoader] = loaders or [PDFLoader(), TextLoader()]

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Every extension any registered loader accepts."""
        return frozenset().union(*(loader.extensions for loader in self._loaders))

    def register(self, loader: PDFLoader | TextLoader) -> None:
        """Add a loader, taking precedence over previously registered ones."""
        self._loaders.insert(0, loader)

    def load(self, path: str) -> list[SourceDocument]:
        """Extract ``path`` using the first loader that claims it.

        Raises:
            UnsupportedFileTypeError: No registered loader handles this type.
        """
        for loader in self._loaders:
            if loader.supports(path):
                return loader.load(path)

        suffix = Path(path).suffix or "(none)"
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. "
            f"Supported types: {', '.join(sorted(self.supported_extensions))}.",
            details={"filename": Path(path).name, "extension": suffix},
        )
