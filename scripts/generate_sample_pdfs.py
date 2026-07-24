"""Generate the sample resume and job description PDFs.

The samples are generated rather than committed as binaries for two reasons: the
text stays reviewable in a diff, and the PDFs are guaranteed to be digital-native
(text extractable) rather than accidental scans — which is precisely the input
class the loader rejects.

Usage::

    python scripts/generate_sample_pdfs.py
    python scripts/generate_sample_pdfs.py --output-dir data/samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tests.fakes import SAMPLE_JOB, SAMPLE_RESUME

DEFAULT_OUTPUT = Path("data/samples")

# Lines matching these are drawn bold as section headings, which is what makes
# the generated PDF resemble a real resume rather than a wall of text — and it
# gives the splitter genuine structural boundaries to work with.
HEADING_MARKERS = (
    "PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "EXPERIENCE", "EDUCATION",
    "CERTIFICATIONS", "REQUIRED QUALIFICATIONS", "PREFERRED QUALIFICATIONS",
    "RESPONSIBILITIES", "JANE DOE", "SENIOR AI ENGINEER",
)


def write_pdf(text: str, path: Path, *, title: str) -> None:
    """Render plain text to a paginated PDF.

    Raises:
        SystemExit: reportlab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
    except ImportError:  # pragma: no cover - environment guard
        raise SystemExit(
            "reportlab is required to generate sample PDFs. Install it with:\n"
            "    pip install reportlab"
        ) from None

    width, height = LETTER
    margin = inch
    line_height = 14
    max_width = width - 2 * margin

    pdf = canvas.Canvas(str(path), pagesize=LETTER)
    pdf.setTitle(title)
    y = height - margin

    for raw_line in text.strip().split("\n"):
        line = raw_line.rstrip()

        if not line:
            y -= line_height / 2
            continue

        is_heading = any(line.strip().startswith(marker) for marker in HEADING_MARKERS)
        pdf.setFont("Helvetica-Bold" if is_heading else "Helvetica", 12 if is_heading else 10)

        for wrapped in wrap(line, pdf, max_width, is_heading):
            if y < margin:
                pdf.showPage()
                pdf.setFont(
                    "Helvetica-Bold" if is_heading else "Helvetica", 12 if is_heading else 10
                )
                y = height - margin
            pdf.drawString(margin, y, wrapped)
            y -= line_height

    pdf.save()


def wrap(line: str, pdf: object, max_width: float, is_heading: bool) -> list[str]:
    """Word-wrap a line to the page width, measured in the actual font."""
    font = "Helvetica-Bold" if is_heading else "Helvetica"
    size = 12 if is_heading else 10

    words = line.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdf.stringWidth(candidate, font, size) <= max_width:  # type: ignore[attr-defined]
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Generate sample PDFs for the analyzer.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = (
        (SAMPLE_RESUME, args.output_dir / "sample_resume.pdf", "Jane Doe - Resume"),
        (SAMPLE_JOB, args.output_dir / "sample_job_description.pdf", "Senior AI Engineer"),
    )
    for text, path, title in targets:
        write_pdf(text, path, title=title)
        print(f"wrote {path} ({path.stat().st_size:,} bytes)")

    # Text copies as well: they exercise the non-PDF ingestion path and make the
    # sample content greppable without opening a PDF viewer.
    for text, path in (
        (SAMPLE_RESUME, args.output_dir / "sample_resume.txt"),
        (SAMPLE_JOB, args.output_dir / "sample_job_description.txt"),
    ):
        path.write_text(text.strip(), encoding="utf-8")
        print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
