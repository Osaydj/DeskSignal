"""Load and chunk the fixed fictional Markdown knowledge base."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

from insightiq.rag.models import DocumentType, KnowledgeChunk


DEFAULT_KNOWLEDGE_BASE_ROOT = Path(__file__).resolve().parents[2] / "knowledge_base"
DOCUMENT_TYPE_BY_FOLDER: dict[str, DocumentType] = {
    "policies": "policy",
    "products": "product",
    "strategy": "strategy",
}
MAX_CHUNK_CHARACTERS = 1_000


def _split_long_section(text: str) -> list[str]:
    """Split long sections at paragraph boundaries without losing text."""

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
    pieces = [
        piece
        for paragraph in paragraphs
        for piece in textwrap.wrap(
            paragraph,
            width=MAX_CHUNK_CHARACTERS,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
    ]
    chunks: list[str] = []
    current = ""

    for paragraph in pieces:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= MAX_CHUNK_CHARACTERS:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def _parse_markdown_document(
    path: Path,
    root: Path,
    document_type: DocumentType,
) -> list[KnowledgeChunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    document_title = path.stem.replace("_", " ").title()
    section_title = "Overview"
    section_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    def finish_section() -> None:
        text = "\n".join(section_lines).strip()
        if text:
            sections.append((section_title, text))

    for line in lines:
        if line.startswith("# "):
            document_title = line[2:].strip() or document_title
            continue
        if line.startswith("## "):
            finish_section()
            section_title = line[3:].strip() or "Untitled section"
            section_lines = []
            continue
        section_lines.append(line)
    finish_section()

    relative_path = path.relative_to(root).as_posix()
    chunks: list[KnowledgeChunk] = []
    for section_index, (heading, section_text) in enumerate(sections, start=1):
        for part_index, part in enumerate(_split_long_section(section_text), start=1):
            identity = f"{relative_path}:{section_index}:{part_index}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"kb.{digest}",
                    document_type=document_type,
                    document_title=document_title,
                    section_title=heading,
                    source_path=relative_path,
                    text=part,
                )
            )
    return chunks


def load_knowledge_base(
    root: Path = DEFAULT_KNOWLEDGE_BASE_ROOT,
) -> list[KnowledgeChunk]:
    """Load only Markdown documents located under approved knowledge folders."""

    resolved_root = root.resolve()
    if not resolved_root.exists():
        return []

    chunks: list[KnowledgeChunk] = []
    for folder_name, document_type in DOCUMENT_TYPE_BY_FOLDER.items():
        folder = (resolved_root / folder_name).resolve()
        if not folder.is_relative_to(resolved_root) or not folder.exists():
            continue
        for path in sorted(folder.glob("*.md")):
            resolved_path = path.resolve()
            if not resolved_path.is_relative_to(folder):
                continue
            chunks.extend(
                _parse_markdown_document(
                    resolved_path,
                    resolved_root,
                    document_type,
                )
            )
    return chunks
