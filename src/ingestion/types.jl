# Data types shared across the ingestion pipeline.

"""
    SourceDocument

A single ingested document with provenance.

# Fields
- `source::String`: Logical source name (e.g. `"FunSQL.jl"`, `"web-search"`).
- `url::String`: URL the content came from.
- `title::String`: Human-readable title (best effort; may be empty).
- `content::String`: Cleaned plain-text content.
"""
struct SourceDocument
    source::String
    url::String
    title::String
    content::String
end

"""
    SearchResult

A single web-search hit returned by an [`AbstractSearchProvider`](@ref).

# Fields
- `title::String`: Result title.
- `url::String`: Result URL.
- `content::String`: Snippet or extracted content (may be empty depending on provider).
- `score::Float64`: Provider-supplied relevance score, or `NaN` if unavailable.
"""
struct SearchResult
    title::String
    url::String
    content::String
    score::Float64
end
