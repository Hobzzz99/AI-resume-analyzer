"""Document loading tests.

Extraction is where real-world failures happen, so the failure paths get as much
coverage as the happy path — every one of them must produce a distinct, typed
error rather than a generic crash (FR-004, FR-031).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.loaders import LoaderRegistry, RawTextLoader, TextLoader
from app.utils.exceptions import EmptyDocumentError, InvalidDocumentError, UnsupportedFileTypeError


class TestTextLoader:
    def test_loads_a_text_file(self, tmp_path: Path) -> None:
        path = tmp_path / "resume.txt"
        path.write_text("Machine learning engineer with six years of experience.", encoding="utf-8")

        documents = TextLoader().load(str(path))

        assert len(documents) == 1
        assert "Machine learning engineer" in documents[0].text
        assert documents[0].filename == "resume.txt"

    def test_unpaginated_sources_report_page_zero(self, tmp_path: Path) -> None:
        """Inventing 'page 1' would let a citation claim precision the source lacks."""
        path = tmp_path / "notes.md"
        path.write_text("# Notes\nSome content here.", encoding="utf-8")
        assert TextLoader().load(str(path))[0].page == 0

    def test_rejects_an_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("   \n  ", encoding="utf-8")

        with pytest.raises(EmptyDocumentError) as caught:
            TextLoader().load(str(path))
        assert "empty" in str(caught.value).lower()

    def test_missing_file_raises_invalid_document(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidDocumentError):
            TextLoader().load(str(tmp_path / "nope.txt"))

    def test_tolerates_bad_bytes(self, tmp_path: Path) -> None:
        """One bad byte should not fail an otherwise readable resume."""
        path = tmp_path / "odd.txt"
        path.write_bytes(b"Python engineer \xff\xfe with experience")
        assert "Python engineer" in TextLoader().load(str(path))[0].text

    def test_supports_reports_its_extensions(self) -> None:
        loader = TextLoader()
        assert loader.supports("a.txt")
        assert loader.supports("A.MD")
        assert not loader.supports("a.pdf")


class TestRawTextLoader:
    def test_wraps_a_string_as_a_document(self) -> None:
        documents = RawTextLoader("Senior AI Engineer role", filename="job.txt").load()
        assert documents[0].text == "Senior AI Engineer role"
        assert documents[0].filename == "job.txt"

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(EmptyDocumentError):
            RawTextLoader("   ").load()


class TestLoaderRegistry:
    def test_dispatches_to_the_matching_loader(self, tmp_path: Path) -> None:
        path = tmp_path / "resume.txt"
        path.write_text("Content about engineering work.", encoding="utf-8")
        assert LoaderRegistry().load(str(path))[0].filename == "resume.txt"

    def test_rejects_an_unsupported_type(self, tmp_path: Path) -> None:
        path = tmp_path / "resume.docx"
        path.write_bytes(b"binary")

        with pytest.raises(UnsupportedFileTypeError) as caught:
            LoaderRegistry().load(str(path))
        assert ".docx" in str(caught.value)

    def test_reports_supported_extensions(self) -> None:
        extensions = LoaderRegistry().supported_extensions
        assert {".pdf", ".txt", ".md"} <= extensions

    def test_a_registered_loader_takes_precedence(self, tmp_path: Path) -> None:
        """New formats are added by registration, with no pipeline change."""

        class UppercasingLoader(TextLoader):
            extensions = frozenset({".txt"})

            def load(self, path: str):  # type: ignore[no-untyped-def]
                documents = super().load(path)
                return [doc.model_copy(update={"text": doc.text.upper()}) for doc in documents]

        path = tmp_path / "resume.txt"
        path.write_text("engineering content here", encoding="utf-8")

        registry = LoaderRegistry()
        registry.register(UppercasingLoader())
        assert registry.load(str(path))[0].text == "ENGINEERING CONTENT HERE"


@pytest.mark.integration
class TestPDFLoader:
    """Requires langchain-community and pypdf; excluded from the default run."""

    def test_reports_pages_one_based(self, tmp_path: Path) -> None:
        from app.rag.loaders import PDFLoader

        pytest.importorskip("reportlab")
        from reportlab.pdfgen import canvas

        path = tmp_path / "two_pages.pdf"
        pdf = canvas.Canvas(str(path))
        pdf.drawString(72, 720, "Page one content about machine learning engineering.")
        pdf.showPage()
        pdf.drawString(72, 720, "Page two content about cloud infrastructure work.")
        pdf.save()

        documents = PDFLoader().load(str(path))
        assert [doc.page for doc in documents] == [1, 2]

    def test_rejects_a_corrupt_pdf(self, tmp_path: Path) -> None:
        from app.rag.loaders import PDFLoader

        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4 this is not actually a pdf")

        with pytest.raises((InvalidDocumentError, EmptyDocumentError)):
            PDFLoader().load(str(path))
