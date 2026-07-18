"""
    Ingestion

Pull documentation into a RAG index from two kinds of sources:

1. **Curated docs** — a fixed, reviewed list of high-quality pages
   (JuliaHealth, OMOP CDM, OHDSI, FunSQL.jl). See [`CURATED_SOURCES`](@ref).
2. **Web search** — a live query against a pluggable search backend for
   anything not in the curated set. See [`AbstractSearchProvider`](@ref).

## Data flow

    CURATED_SOURCES ─┐
                     ├─▶ fetch_url ─▶ SourceDocument ─┐
    web_search ──────┘                                ├─▶ ingest ─▶ ingest_to_index ─▶ RAG index
                                                       │
                          (each hit's URL fetched) ────┘

## Layout (one concern per file, in include order)

| File                      | Responsibility                                   |
|---------------------------|--------------------------------------------------|
| `ingestion/types.jl`      | `SourceDocument`, `SearchResult` data types      |
| `ingestion/curated.jl`    | `CURATED_SOURCES` registry                        |
| `ingestion/fetch.jl`      | HTTP fetch + HTML→text (`fetch_url`, `html_to_text`, `fetch_curated`) |
| `ingestion/search.jl`     | Search providers (`web_search`, `DuckDuckGoProvider`) |
| `ingestion/pipeline.jl`   | Orchestration (`ingest`, `ingest_to_index`)      |

## Quickstart

```julia
using HealthLLM

register_models("llama3.2", "nomic-embed-text")
index = ingest_to_index(; sources=["FunSQL.jl", "OMOP CDM"],
                          query="OMOP condition_occurrence table",
                          embedder_kwargs=(model="nomic-embed-text",))
answer = generate_funsql_query(index, "nomic-embed-text", "llama3.2",
                               "Context: {input_query}", "How do I join person and visit?")
```
"""
module Ingestion

using HTTP
using URIs
using RAGTools
using ..Utils

export SourceDocument, SearchResult,
    AbstractSearchProvider, DuckDuckGoProvider,
    default_search_provider, web_search,
    CURATED_SOURCES, fetch_url, html_to_text, fetch_curated,
    Chunk, AbstractChunkStrategy, RecursiveChunk, HeaderChunk, RecordChunk, FixedSizeChunk,
    chunk, chunk_document, chunk_provenance, default_strategy, load_funsql_examples,
    ingest, ingest_to_index

# Included in dependency order: later files use the types/functions above them.
include("ingestion/types.jl")
include("ingestion/curated.jl")
include("ingestion/fetch.jl")
include("ingestion/search.jl")
include("ingestion/chunk.jl")
include("ingestion/pipeline.jl")

end
