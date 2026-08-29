"""Small dependency-free TF-IDF retriever for the fictional knowledge base."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from desksignal.rag.loader import DEFAULT_KNOWLEDGE_BASE_ROOT, load_knowledge_base
from desksignal.rag.models import DocumentType, KnowledgeChunk, RetrievalResult


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "anything",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "in",
        "is",
        "it",
        "might",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "there",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "with",
    }
)
TOKEN_NORMALIZATION = {
    "sale": "revenue",
    "sales": "revenue",
    "sold": "revenue",
    "drop": "decline",
    "dropped": "decline",
    "dropping": "decline",
    "decreased": "decline",
    "decrease": "decline",
    "declined": "decline",
    "fall": "decline",
    "fell": "decline",
    "products": "product",
    "promotions": "promotion",
    "documents": "document",
    "policies": "policy",
    "strategies": "strategy",
}


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        normalized = TOKEN_NORMALIZATION.get(token, token)
        if normalized not in STOP_WORDS and len(normalized) > 1:
            tokens.append(normalized)
    return tokens


def _tf_idf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    return {
        token: (1 + math.log(count)) * idf[token]
        for token, count in counts.items()
        if token in idf
    }


def _cosine_similarity(first: dict[str, float], second: dict[str, float]) -> float:
    common = first.keys() & second.keys()
    numerator = sum(first[token] * second[token] for token in common)
    first_norm = math.sqrt(sum(value * value for value in first.values()))
    second_norm = math.sqrt(sum(value * value for value in second.values()))
    if not first_norm or not second_norm:
        return 0.0
    return numerator / (first_norm * second_norm)


def _searchable_text(chunk: KnowledgeChunk) -> str:
    return (
        f"{chunk.document_title} {chunk.section_title} "
        f"{chunk.section_title} {chunk.text}"
    )


def search_knowledge_base(
    query: str,
    *,
    top_k: int = 3,
    document_type: DocumentType | None = None,
    root: Path = DEFAULT_KNOWLEDGE_BASE_ROOT,
) -> list[RetrievalResult]:
    """Return only the highest-scoring relevant knowledge-base sections."""

    if not 1 <= top_k <= 5:
        raise ValueError("top_k must be between 1 and 5.")

    chunks = load_knowledge_base(root)
    if document_type is not None:
        chunks = [chunk for chunk in chunks if chunk.document_type == document_type]
    if not chunks:
        return []

    chunk_tokens = [_tokenize(_searchable_text(chunk)) for chunk in chunks]
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    document_frequency = Counter(
        token for tokens in chunk_tokens for token in set(tokens)
    )
    idf = {
        token: math.log((1 + len(chunks)) / (1 + frequency)) + 1
        for token, frequency in document_frequency.items()
    }
    query_vector = _tf_idf_vector(query_tokens, idf)

    scored: list[RetrievalResult] = []
    for chunk, tokens in zip(chunks, chunk_tokens, strict=True):
        score = _cosine_similarity(query_vector, _tf_idf_vector(tokens, idf))
        if score > 0:
            scored.append(RetrievalResult(chunk=chunk, score=round(score, 6)))

    return sorted(
        scored,
        key=lambda result: (-result.score, result.chunk.source_path, result.chunk.chunk_id),
    )[:top_k]


def knowledge_base_metadata(
    root: Path = DEFAULT_KNOWLEDGE_BASE_ROOT,
) -> dict[str, object]:
    """Return document titles and types without sending document contents."""

    chunks = load_knowledge_base(root)
    documents = {
        (chunk.source_path, chunk.document_title, chunk.document_type)
        for chunk in chunks
    }
    return {
        "document_count": len(documents),
        "section_count": len(chunks),
        "documents": [
            {"source_path": path, "title": title, "type": document_type}
            for path, title, document_type in sorted(documents)
        ],
    }
