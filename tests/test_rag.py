"""Tests for the local fictional-company retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

from desksignal.rag.loader import MAX_CHUNK_CHARACTERS, load_knowledge_base
from desksignal.rag.retriever import knowledge_base_metadata, search_knowledge_base


def test_loader_chunks_only_approved_fictional_documents() -> None:
    chunks = load_knowledge_base()
    metadata = knowledge_base_metadata()

    assert metadata["document_count"] == 4
    assert metadata["section_count"] == len(chunks)
    assert {chunk.document_type for chunk in chunks} == {
        "policy",
        "product",
        "strategy",
    }
    assert all(len(chunk.text) <= MAX_CHUNK_CHARACTERS for chunk in chunks)
    assert all(
        chunk.source_path.startswith(("policies/", "products/", "strategy/"))
        for chunk in chunks
    )


def test_retrieval_ranks_the_relevant_product_section_first() -> None:
    results = search_knowledge_base(
        "HD Webcam July revenue decline promotion marketing placement",
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].chunk.document_title == (
        "Northstar Office Co. Product Portfolio Notes"
    )
    assert results[0].chunk.section_title == "HD Webcam"
    assert all(result.score > 0 for result in results)


def test_document_type_filter_and_irrelevant_query() -> None:
    policy_results = search_knowledge_base(
        "Office Furniture profit margin discount policy",
        top_k=2,
        document_type="policy",
    )

    assert policy_results
    assert all(result.chunk.document_type == "policy" for result in policy_results)
    assert search_knowledge_base("quasar nebula astronomy", top_k=3) == []


def test_oversized_markdown_section_is_split_safely(tmp_path: Path) -> None:
    policies_folder = tmp_path / "policies"
    policies_folder.mkdir()
    (policies_folder / "long_policy.md").write_text(
        "# Long Policy\n\n## One Section\n\n" + ("fictional policy text " * 200),
        encoding="utf-8",
    )

    chunks = load_knowledge_base(tmp_path)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= MAX_CHUNK_CHARACTERS for chunk in chunks)
