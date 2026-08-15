"""Paper formatting to multiple output formats (LaTeX, PDF, DOCX, JSON)."""

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import pypandoc
except ImportError:
    pypandoc = None

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

from research_machine.schemas import DraftPaper

logger = logging.getLogger(__name__)


class PaperFormatter:
    """Formats papers into multiple output formats."""

    LATEX_TEMPLATE = r"""
\documentclass[11pt, a4paper]{article}
\usepackage[utf-8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[numbers]{natbib}
\usepackage{setspace}

\onehalfspacing

\title{TITLE_PLACEHOLDER}
\author{AUTHOR_PLACEHOLDER}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
ABSTRACT_PLACEHOLDER
\end{abstract}

CONTENT_PLACEHOLDER

\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
"""

    def __init__(self):
        self.verify_pandoc()

    @staticmethod
    def verify_pandoc():
        """Check if pandoc is installed."""
        if pypandoc is None:
            logger.warning("pypandoc not installed; LaTeX/DOCX conversion may fail")
            return

        try:
            pypandoc.get_pandoc_version()
            logger.debug("Pandoc is available")
        except Exception as e:
            logger.warning(f"Pandoc verification failed: {e}")

    async def to_latex(self, draft: DraftPaper) -> str:
        """
        Convert markdown paper to LaTeX.

        Args:
            draft: DraftPaper object

        Returns:
            LaTeX source code as string
        """
        try:
            # Build LaTeX content from sections
            content_parts = [
                f"\\section{{Introduction}}\n{draft.introduction_md}",
                f"\\section{{Methods}}\n{draft.methods_md}",
                f"\\section{{Results}}\n{draft.results_md}",
                f"\\section{{Discussion}}\n{draft.discussion_md}",
            ]

            content = "\n\n".join(content_parts)

            # Create LaTeX document
            latex_doc = self.LATEX_TEMPLATE
            latex_doc = latex_doc.replace('TITLE_PLACEHOLDER', self._escape_latex(draft.title))
            latex_doc = latex_doc.replace('AUTHOR_PLACEHOLDER', 'Automatic Research Machine')
            latex_doc = latex_doc.replace('ABSTRACT_PLACEHOLDER', self._escape_latex(draft.abstract))
            latex_doc = latex_doc.replace('CONTENT_PLACEHOLDER', content)

            logger.debug(f"Generated LaTeX document for '{draft.title}'")
            return latex_doc

        except Exception as e:
            logger.error(f"Error converting to LaTeX: {e}")
            raise

    async def to_pdf(self, latex_content: str) -> bytes:
        """
        Convert LaTeX to PDF via pdflatex.

        Args:
            latex_content: LaTeX source code

        Returns:
            PDF as bytes

        Note:
            Requires pdflatex to be installed on system.
            Falls back to pandoc if available.
        """
        try:
            # Try using pdflatex directly
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as tex_file:
                tex_file.write(latex_content)
                tex_path = tex_file.name

            try:
                # Run pdflatex
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-output-directory=/tmp', tex_path],
                    capture_output=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    pdf_path = Path(tex_path).with_suffix('.pdf')
                    if pdf_path.exists():
                        with open(pdf_path, 'rb') as f:
                            pdf_bytes = f.read()
                        logger.debug(f"Generated PDF ({len(pdf_bytes)} bytes)")
                        return pdf_bytes

                logger.warning("pdflatex failed; attempting fallback with pandoc")

            finally:
                # Clean up temp files
                Path(tex_path).unlink(missing_ok=True)

            # Fallback: use pandoc to convert LaTeX to PDF
            if pypandoc:
                pdf_bytes = pypandoc.convert_text(
                    latex_content,
                    'pdf',
                    format='latex',
                )
                if pdf_bytes:
                    logger.debug(f"Generated PDF via pandoc ({len(pdf_bytes)} bytes)")
                    return pdf_bytes

            logger.error("PDF generation failed; neither pdflatex nor pandoc available")
            raise RuntimeError("No PDF generator available")

        except subprocess.TimeoutExpired:
            logger.error("pdflatex timed out")
            raise
        except Exception as e:
            logger.error(f"Error converting LaTeX to PDF: {e}")
            raise

    async def to_docx(self, draft: DraftPaper) -> bytes:
        """
        Convert paper to DOCX format.

        Args:
            draft: DraftPaper object

        Returns:
            DOCX as bytes
        """
        try:
            doc = Document()

            # Title
            title = doc.add_heading(draft.title, level=0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER

            # Abstract
            doc.add_heading('Abstract', level=1)
            doc.add_paragraph(draft.abstract)

            # Introduction
            doc.add_heading('Introduction', level=1)
            doc.add_paragraph(draft.introduction_md)

            # Methods
            doc.add_heading('Methods', level=1)
            doc.add_paragraph(draft.methods_md)

            # Results
            doc.add_heading('Results', level=1)
            doc.add_paragraph(draft.results_md)

            # Discussion
            doc.add_heading('Discussion', level=1)
            doc.add_paragraph(draft.discussion_md)

            # Save to bytes
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
                doc.save(f.name)
                with open(f.name, 'rb') as docx_file:
                    docx_bytes = docx_file.read()
                Path(f.name).unlink()

            logger.debug(f"Generated DOCX ({len(docx_bytes)} bytes)")
            return docx_bytes

        except Exception as e:
            logger.error(f"Error converting to DOCX: {e}")
            raise

    async def to_json(
        self,
        draft: DraftPaper,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert paper to JSON format with full metadata.

        Args:
            draft: DraftPaper object
            metadata: Additional metadata (quality scores, timestamps, etc.)

        Returns:
            JSON-serializable dict
        """
        try:
            paper_json = {
                "metadata": {
                    "title": draft.title,
                    "generated_at": datetime.utcnow().isoformat(),
                    "format_version": "1.0",
                    **(metadata or {}),
                },
                "abstract": draft.abstract,
                "sections": {
                    "introduction": draft.introduction_md,
                    "methods": draft.methods_md,
                    "results": draft.results_md,
                    "discussion": draft.discussion_md,
                },
                "citations": {
                    "count": draft.citation_count or 0,
                    "references": [
                        {
                            "id": idx + 1,
                            "title": c.get("title") if isinstance(c, dict) else getattr(c, "title", None),
                            "year": c.get("year") if isinstance(c, dict) else getattr(c, "year", None),
                            "authors": c.get("authors") if isinstance(c, dict) else getattr(c, "authors", None),
                        }
                        for idx, c in enumerate(getattr(draft, "citations", None) or [])
                    ],
                },
                "full_content": {
                    "markdown": getattr(draft, "content_markdown", ""),
                    "latex": getattr(draft, "content_latex", ""),
                },
            }

            logger.debug(f"Generated JSON metadata for '{draft.title}'")
            return paper_json

        except Exception as e:
            logger.error(f"Error converting to JSON: {e}")
            raise

    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape special characters for LaTeX."""
        if not text:
            return ""

        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
        }

        result = text
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)

        return result

    async def save_all_formats(
        self,
        draft: DraftPaper,
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Save paper in all supported formats.

        Args:
            draft: DraftPaper object
            output_dir: Output directory (will be created if needed)
            metadata: Additional metadata to include

        Returns:
            Dict mapping format name to file path
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        try:
            # Markdown (original)
            md_path = output_dir / "paper.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(draft.content_markdown or "")
            outputs['markdown'] = md_path
            logger.info(f"Saved markdown: {md_path}")

        except Exception as e:
            logger.error(f"Error saving markdown: {e}")

        try:
            # LaTeX
            latex_content = await self.to_latex(draft)
            tex_path = output_dir / "paper.tex"
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            outputs['latex'] = tex_path
            logger.info(f"Saved LaTeX: {tex_path}")

        except Exception as e:
            logger.error(f"Error saving LaTeX: {e}")

        try:
            # PDF (from LaTeX)
            if 'latex' in outputs:
                latex_content = await self.to_latex(draft)
                pdf_bytes = await self.to_pdf(latex_content)
                pdf_path = output_dir / "paper.pdf"
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_bytes)
                outputs['pdf'] = pdf_path
                logger.info(f"Saved PDF: {pdf_path}")

        except Exception as e:
            logger.warning(f"Error saving PDF: {e}")

        try:
            # DOCX
            docx_bytes = await self.to_docx(draft)
            docx_path = output_dir / "paper.docx"
            with open(docx_path, 'wb') as f:
                f.write(docx_bytes)
            outputs['docx'] = docx_path
            logger.info(f"Saved DOCX: {docx_path}")

        except Exception as e:
            logger.error(f"Error saving DOCX: {e}")

        try:
            # JSON metadata
            json_data = await self.to_json(draft, metadata)
            json_path = output_dir / "metadata.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            outputs['json'] = json_path
            logger.info(f"Saved JSON: {json_path}")

        except Exception as e:
            logger.error(f"Error saving JSON: {e}")

        return outputs
