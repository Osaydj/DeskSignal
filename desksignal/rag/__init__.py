"""Local retrieval components for the fictional company knowledge base."""

from desksignal.rag.loader import DEFAULT_KNOWLEDGE_BASE_ROOT, load_knowledge_base
from desksignal.rag.retriever import knowledge_base_metadata, search_knowledge_base

__all__ = [
    "DEFAULT_KNOWLEDGE_BASE_ROOT",
    "knowledge_base_metadata",
    "load_knowledge_base",
    "search_knowledge_base",
]
