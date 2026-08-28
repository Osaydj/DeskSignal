"""Data models for knowledge-base chunks and retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DocumentType = Literal["policy", "product", "strategy"]


@dataclass(frozen=True)
class KnowledgeChunk:
    """One retrievable Markdown section with source metadata."""

    chunk_id: str
    document_type: DocumentType
    document_title: str
    section_title: str
    source_path: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    """One chunk plus its local lexical relevance score."""

    chunk: KnowledgeChunk
    score: float
