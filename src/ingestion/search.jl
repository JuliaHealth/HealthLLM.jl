# Pluggable web-search backends. One provider is wired up today (DuckDuckGo);
# add another by defining a struct <: AbstractSearchProvider and a `web_search`
# method for it — nothing else in the pipeline needs to change.

"""
    AbstractSearchProvider

Interface for web-search backends. Implement [`web_search`](@ref) for a concrete
provider to plug it into ingestion. Only [`DuckDuckGoProvider`](@ref) is wired up
for now; add more providers as subtypes later.
"""
abstract type AbstractSearchProvider end

"""
    DuckDuckGoProvider()

Keyless web search via the DuckDuckGo HTML endpoint. No API key required.
The endpoint is rate-limited and results are best-effort scraped from the HTML.
"""
struct DuckDuckGoProvider <: AbstractSearchProvider end

"""
    default_search_provider() -> AbstractSearchProvider

Return the default search backend. Currently [`DuckDuckGoProvider`](@ref) (keyless).
"""
default_search_provider() = DuckDuckGoProvider()

"""
    web_search(query; max_results=5) -> Vector{SearchResult}
    web_search(provider, query; max_results=5) -> Vector{SearchResult}

Run `query` against `provider` and return up to `max_results` [`SearchResult`](@ref)s.
`provider` defaults to [`default_search_provider`](@ref).

# Example

```julia
hits = web_search("OMOP CDM person table columns"; max_results=3)
```
"""
web_search(query::AbstractString; kwargs...) = web_search(default_search_provider(), query; kwargs...)

function web_search(::DuckDuckGoProvider, query::AbstractString; max_results::Integer=5)
    url = string("https://html.duckduckgo.com/html/?q=", URIs.escapeuri(String(query)))
    resp = HTTP.get(url; headers=_DEFAULT_HEADERS, readtimeout=30, status_exception=true)
    body = String(resp.body)
    results = SearchResult[]
    # Each hit is an <a class="result__a" href="...">title</a> element.
    for m in eachmatch(r"class=\"result__a\"[^>]*href=\"(.*?)\"[^>]*>(.*?)</a>"s, body)
        href, title = m[1], html_to_text(m[2])
        # DuckDuckGo wraps the real destination in a redirect: .../l/?uddg=<encoded-url>.
        real = href
        rm = match(r"uddg=([^&]+)", href)
        rm !== nothing && (real = URIs.unescapeuri(rm[1]))
        startswith(real, "//") && (real = "https:" * real)
        push!(results, SearchResult(String(title), String(real), "", NaN))
        length(results) >= max_results && break
    end
    return results
end
