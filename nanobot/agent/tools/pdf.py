"""PDF reading tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem import _resolve_path


class ReadPDFTool(Tool):
    """Extract text content from PDF files."""

    _MAX_CHARS = 100_000  # Max characters to return

    @property
    def name(self) -> str:
        return "read_pdf"

    @property
    def description(self) -> str:
        return (
            "Extract text content from a PDF file. "
            "Supports pagination via page_start and page_end parameters. "
            "Returns extracted text with page markers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file",
                },
                "page_start": {
                    "type": "integer",
                    "description": "First page to read (1-indexed, default: 1)",
                    "minimum": 1,
                },
                "page_end": {
                    "type": "integer",
                    "description": "Last page to read (inclusive, default: all pages)",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
        extra_allowed_dirs: list[Path] | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        self._extra_allowed_dirs = extra_allowed_dirs

    async def execute(
        self,
        path: str,
        page_start: int = 1,
        page_end: int | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            import pdfplumber
        except ImportError:
            return (
                "Error: pdfplumber not installed. "
                "Install with: pip install pdfplumber"
            )

        try:
            fp = _resolve_path(
                path, self._workspace, self._allowed_dir, self._extra_allowed_dirs
            )

            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"
            if not path.lower().endswith(".pdf"):
                return f"Error: File is not a PDF: {path}"

            with pdfplumber.open(fp) as pdf:
                total_pages = len(pdf.pages)

                if total_pages == 0:
                    return f"PDF is empty: {path}"

                # Adjust page range
                start = max(1, page_start)
                end = min(page_end or total_pages, total_pages)

                if start > end:
                    return (
                        f"Error: page_start ({page_start}) > page_end ({page_end})"
                    )

                if start > total_pages:
                    return (
                        f"Error: page_start ({page_start}) exceeds "
                        f"total pages ({total_pages})"
                    )

                lines = [
                    f"PDF: {fp.name}",
                    f"Total pages: {total_pages}",
                    f"Reading pages {start}-{end}",
                    "=" * 50,
                    "",
                ]

                total_chars = 0
                truncated = False

                for page_num in range(start, end + 1):
                    page = pdf.pages[page_num - 1]
                    text = page.extract_text() or ""

                    # Clean up whitespace
                    text = text.strip()

                    lines.append(f"--- Page {page_num} ---")
                    if text:
                        lines.append(text)
                    else:
                        lines.append("(No extractable text on this page)")
                    lines.append("")

                    total_chars += len(text)
                    if total_chars > self._MAX_CHARS:
                        truncated = True
                        break

                result = "\n".join(lines)

                if truncated:
                    result += (
                        f"\n\n(Content truncated after {self._MAX_CHARS} characters. "
                        f"Use page_start/page_end to read specific pages.)"
                    )

                if end < total_pages and not truncated:
                    result += (
                        f"\n\n(Showing pages {start}-{end} of {total_pages}. "
                        f"Use page_start={end + 1} to continue.)"
                    )

                return result

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.error("Failed to read PDF {}: {}", path, e)
            return f"Error reading PDF: {e}"


class PDFInfoTool(Tool):
    """Get metadata and information about a PDF file."""

    @property
    def name(self) -> str:
        return "pdf_info"

    @property
    def description(self) -> str:
        return (
            "Get metadata and information about a PDF file: "
            "page count, title, author, creation date, etc."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file",
                },
            },
            "required": ["path"],
        }

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
        extra_allowed_dirs: list[Path] | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        self._extra_allowed_dirs = extra_allowed_dirs

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            import pdfplumber
        except ImportError:
            return (
                "Error: pdfplumber not installed. "
                "Install with: pip install pdfplumber"
            )

        try:
            fp = _resolve_path(
                path, self._workspace, self._allowed_dir, self._extra_allowed_dirs
            )

            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"
            if not path.lower().endswith(".pdf"):
                return f"Error: File is not a PDF: {path}"

            with pdfplumber.open(fp) as pdf:
                info = []
                info.append(f"File: {fp.name}")
                info.append(f"Path: {fp}")
                info.append(f"Size: {fp.stat().st_size / 1024:.1f} KB")
                info.append(f"Pages: {len(pdf.pages)}")

                # PDF metadata
                metadata = pdf.metadata
                if metadata:
                    info.append("")
                    info.append("PDF Metadata:")
                    for key, value in metadata.items():
                        if value:
                            # Clean up key name
                            clean_key = key.replace("/", "").strip()
                            info.append(f"  {clean_key}: {value}")

                # Try to extract text from first page for preview
                if pdf.pages:
                    first_page_text = pdf.pages[0].extract_text() or ""
                    first_page_text = first_page_text.strip()
                    if first_page_text:
                        preview = first_page_text[:300]
                        if len(first_page_text) > 300:
                            preview += "..."
                        info.append("")
                        info.append("First page preview:")
                        info.append(preview)

                return "\n".join(info)

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.error("Failed to get PDF info {}: {}", path, e)
            return f"Error reading PDF info: {e}"
