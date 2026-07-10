"""Named pipeline stages for the ETV_V2 refactor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStage:
    id: str
    label: str
    output: str


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage("stage_0", "Import Scopus exports from Query 1-4", "raw query datasets"),
    PipelineStage(
        "stage_0_5",
        "Merge, deduplicate, and preserve query provenance",
        "deduplicated corpus with query_sources",
    ),
    PipelineStage(
        "stage_1",
        "Validate journals/source titles",
        "source-title validated corpus",
    ),
    PipelineStage(
        "stage_1_5",
        "Filter for AI x entrepreneurship relevance",
        "relevance-filtered corpus",
    ),
    PipelineStage(
        "stage_1_6",
        "Create one-hot query columns and query-specific views",
        "master analytical CSV",
    ),
    PipelineStage(
        "stage_1b",
        "Export VOSviewer files for full corpus and Query 1-4 subsets",
        "VOSviewer input files",
    ),
    PipelineStage(
        "stage_2a",
        "Run BERTopic and KeyBERT/keyphrase extraction",
        "topic and keyword assignments",
    ),
    PipelineStage(
        "stage_2a_5",
        "Code AI Specification Framework per paper",
        "specification profiles",
    ),
    PipelineStage(
        "stage_2b",
        "Build knowledge graph for theory elaboration",
        "query-aware specification KG",
    ),
    PipelineStage(
        "stage_3",
        "Serve analytics and visualization",
        "platform views and evidence panels",
    ),
)
