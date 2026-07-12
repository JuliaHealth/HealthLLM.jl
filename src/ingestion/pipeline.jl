# The orchestration layer: gather documents (curated + search) and, optionally,
# turn them straight into a RAG index.

"""
    ingest(; sources=keys(CURATED_SOURCES), query=nothing, provider=default_search_provider(),
             max_results=5, fetch_search_content=true, timeout=30, min_length=200) -> Vector{SourceDocument}

Gather documents from curated sources and/or a live web search into a single
vector of [`SourceDocument`](@ref)s.

# Keywords
- `sources`: Curated source names to fetch (see [`CURATED_SOURCES`](@ref)). Pass
  `String[]`/`nothing` to skip curated fetching and search only.
- `query`: If given, also run a web search for `query` via `provider`.
- `provider`: Search backend (default: [`default_search_provider`](@ref)).
- `max_results`: Max web-search hits to keep.
- `fetch_search_content`: When `true`, fetch each result URL for full text; when
  `false`, use only the provider-supplied snippet/content.
- `timeout`, `min_length`: Passed through to fetching/filtering.

# Example

```julia
# Curated docs plus a targeted web search
docs = ingest(; sources=["FunSQL.jl"], query="FunSQL From Where Select example")
```
"""
function ingest(; sources=keys(CURATED_SOURCES), query::Union{Nothing,AbstractString}=nothing,
    provider::AbstractSearchProvider=default_search_provider(), max_results::Integer=5,
    fetch_search_content::Bool=true, timeout::Real=30, min_length::Integer=200)

    docs = SourceDocument[]

    # 1. Curated docs (skipped when `sources` is empty/nothing).
    if sources !== nothing && !isempty(collect(sources))
        append!(docs, fetch_curated(sources; timeout=timeout, min_length=min_length))
    end

    # 2. Live web search (only when a `query` is given).
    if query !== nothing
        for hit in web_search(provider, query; max_results=max_results)
            content = hit.content
            if fetch_search_content && !isempty(hit.url)
                try
                    content = fetch_url(hit.url; timeout=timeout)
                catch err
                    @warn "Failed to fetch search result, using snippet: $(hit.url) ($err)"
                end
            end
            isempty(strip(content)) && continue
            push!(docs, SourceDocument("web-search", hit.url, hit.title, content))
        end
    end

    return docs
end

"""
    ingest_to_index(cfg=RAGTools.SimpleIndexer(); sources=keys(CURATED_SOURCES), query=nothing,
                    provider=default_search_provider(), max_results=5, fetch_search_content=true,
                    timeout=30, min_length=200, embedder_kwargs=NamedTuple(), docs=nothing) -> index

Ingest curated docs and/or web-search results and build a RAG index directly from
the in-memory text — no intermediate files.

Documents are chunked with `RAGTools.TextChunker()` and their URLs are recorded as
chunk sources for provenance. Pass a precomputed `docs` vector (from [`ingest`](@ref))
to reuse it instead of fetching again. The returned index is ready for
`generate_funsql_query`.

# Example

```julia
register_models("llama3.2", "nomic-embed-text")
index = ingest_to_index(; sources=["FunSQL.jl", "OMOP CDM"],
                          query="OMOP condition_occurrence table",
                          embedder_kwargs=(model="nomic-embed-text",))
answer = generate_funsql_query(index, "nomic-embed-text", "llama3.2",
                               "Context: {input_query}", "How do I join person and visit?")
```
"""
function ingest_to_index(cfg=RAGTools.SimpleIndexer();
    docs::Union{Nothing,AbstractVector{SourceDocument}}=nothing,
    embedder_kwargs=NamedTuple(), kwargs...)

    documents = docs === nothing ? ingest(; kwargs...) : docs
    isempty(documents) &&
        throw(ArgumentError("No documents ingested; nothing to index. Check sources/query and network."))

    contents = [d.content for d in documents]
    provenance = [isempty(d.url) ? d.source : d.url for d in documents]

    return RAGTools.build_index(cfg, contents;
        chunker=RAGTools.TextChunker(),
        chunker_kwargs=(sources=provenance,),
        embedder_kwargs=embedder_kwargs)
end
