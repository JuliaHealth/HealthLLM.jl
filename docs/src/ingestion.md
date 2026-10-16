```@meta
CurrentModule = HealthLLM
```

# Document Ingestion

The ingestion layer pulls documentation into a RAG index from two kinds of
sources and hands the result straight to [`generate_funsql_query`](@ref):

1. **Curated docs** — a fixed, reviewed list of high-quality pages
   (JuliaHealth, OMOP CDM, OHDSI, FunSQL.jl). See [`CURATED_SOURCES`](@ref).
2. **Web search** — a live query against a pluggable search backend for
   anything outside the curated set. See [`AbstractSearchProvider`](@ref).

```
CURATED_SOURCES ─┐
                 ├─▶ fetch_url ─▶ SourceDocument ─┐
web_search ──────┘                                ├─▶ ingest ─▶ ingest_to_index ─▶ RAG index
                                                   │
                      (each hit's URL fetched) ────┘
```

Everything is available from the single `using HealthLLM` entry point.

!!! note "Prerequisites"
    Building an index calls an embedding model, so register one first (for
    example a local [Ollama](https://ollama.com) model with
    `register_models("llama3.2", "nomic-embed-text")`). Fetching and search make
    outbound HTTP requests, so these functions need network access. The default
    search backend ([`DuckDuckGoProvider`](@ref)) is keyless.

## Quick demo

Curated docs plus a targeted web search, indexed and queried end to end:

```julia
using HealthLLM

register_models("llama3.2", "nomic-embed-text")

index = ingest_to_index(;
    sources = ["FunSQL.jl", "OMOP CDM"],
    query   = "OMOP condition_occurrence table columns",
    embedder_kwargs = (model = "nomic-embed-text",),
)

answer = generate_funsql_query(
    index,
    "nomic-embed-text",
    "llama3.2",
    "Context: {input_query}. Answer with a FunSQL query.",
    "How do I join the person and visit_occurrence tables?",
)
```

The rest of this page tours each building block so you can compose your own
pipeline.

## The building blocks

### 1. Curated sources

[`CURATED_SOURCES`](@ref) is a registry mapping a source name to a list of
candidate URLs. Raw Markdown endpoints are preferred; rendered HTML pages are
accepted and cleaned to text automatically.

```julia
julia> collect(keys(CURATED_SOURCES))
4-element Vector{String}:
 "JuliaHealth"
 "OMOP CDM"
 "OHDSI"
 "FunSQL.jl"

julia> CURATED_SOURCES["FunSQL.jl"]
3-element Vector{String}:
 "https://raw.githubusercontent.com/MechanicalRabbit/FunSQL.jl/master/docs/src/index.md"
 "https://raw.githubusercontent.com/MechanicalRabbit/FunSQL.jl/master/docs/src/guide/index.md"
 "https://raw.githubusercontent.com/MechanicalRabbit/FunSQL.jl/master/docs/src/examples/index.md"
```

Add your own source by pushing to the registry before ingesting:

```julia
CURATED_SOURCES["MyDocs"] = ["https://example.org/docs/index.md"]
```

### 2. Fetching and cleaning a URL

[`fetch_url`](@ref) downloads a URL and returns clean text. HTML is stripped to
readable text with [`html_to_text`](@ref); Markdown and plain text pass through
untouched.

```julia
text = fetch_url("https://raw.githubusercontent.com/OHDSI/CommonDataModel/main/README.md")

html_to_text("<p>Hello <b>world</b></p>")   # -> "Hello world"
```

### 3. Fetching the curated set

[`fetch_curated`](@ref) walks the registry, tries each candidate URL, and returns
a flat vector of [`SourceDocument`](@ref)s. URLs that fail (404, timeout) are
skipped with a warning, so partial results still come through.

```julia
docs = fetch_curated(["FunSQL.jl", "OMOP CDM"])

for d in docs
    println(d.source, "  <-  ", d.url, "  (", length(d.content), " chars)")
end
```

Each [`SourceDocument`](@ref) carries its `source`, `url`, `title`, and cleaned
`content`.

### 4. Web search

[`web_search`](@ref) runs a query against a search backend and returns
[`SearchResult`](@ref)s. With no provider argument it uses
[`default_search_provider`](@ref) (currently the keyless
[`DuckDuckGoProvider`](@ref)).

```julia
hits = web_search("OMOP CDM person table columns"; max_results=3)

for h in hits
    println(h.title, "  =>  ", h.url)
end
```

Search is pluggable: define a `struct MyProvider <: AbstractSearchProvider` and a
`web_search(::MyProvider, query; max_results)` method to add a backend — nothing
else in the pipeline changes.

### 5. Gathering documents with `ingest`

[`ingest`](@ref) combines curated fetching and web search into one
`Vector{SourceDocument}`. Use `sources` for curated names and `query` for a live
search; either can be omitted.

```julia
# Curated only
docs = ingest(; sources = ["FunSQL.jl"])

# Search only (skip curated docs)
docs = ingest(; sources = String[], query = "FunSQL From Where Select example")

# Both
docs = ingest(; sources = ["FunSQL.jl"], query = "FunSQL aggregate group example")
```

By default each search hit's page is fetched for full text
(`fetch_search_content = true`); set it to `false` to keep only the provider's
snippet.

### 6. Building an index with `ingest_to_index`

[`ingest_to_index`](@ref) ingests documents and builds a RAG index directly from
the in-memory text — no intermediate files. Each chunk keeps its origin URL as
its source for provenance.

```julia
register_models("llama3.2", "nomic-embed-text")

index = ingest_to_index(;
    sources = ["FunSQL.jl", "OMOP CDM"],
    query   = "OMOP condition_occurrence table",
    embedder_kwargs = (model = "nomic-embed-text",),
)
```

If you already have documents from [`ingest`](@ref), reuse them instead of
fetching again:

```julia
docs  = ingest(; sources = ["OHDSI"])
index = ingest_to_index(; docs = docs, embedder_kwargs = (model = "nomic-embed-text",))
```

## Full end-to-end demo

Save this as `ingestion_demo.jl` and run it with `julia --project=. ingestion_demo.jl`
(requires network access and a running embedding model):

```julia
using HealthLLM

# 1. Register the chat + embedding models (here: local Ollama models).
register_models("llama3.2", "nomic-embed-text")

# 2. Inspect what curated sources are available.
println("Curated sources: ", collect(keys(CURATED_SOURCES)))

# 3. Gather documents: curated FunSQL + OMOP docs, plus a live web search.
docs = ingest(;
    sources = ["FunSQL.jl", "OMOP CDM"],
    query   = "OMOP CDM condition_occurrence table definition",
    max_results = 3,
)
println("\nGathered $(length(docs)) documents:")
for d in docs
    println("  [$(d.source)] $(first(d.title, 60))  ($(length(d.content)) chars)")
end

# 4. Build a RAG index straight from the gathered text.
index = ingest_to_index(; docs = docs, embedder_kwargs = (model = "nomic-embed-text",))

# 5. Ask a question against the freshly ingested docs.
answer = generate_funsql_query(
    index,
    "nomic-embed-text",
    "llama3.2",
    "Context: {input_query}. Answer with a concise FunSQL query.",
    "Write a FunSQL query that counts conditions per person.",
)
println("\nAnswer:\n", answer)
```

Expected shape of the output (content varies with the live sources and model):

```
Curated sources: ["JuliaHealth", "OMOP CDM", "OHDSI", "FunSQL.jl"]

Gathered 8 documents:
  [FunSQL.jl] FunSQL.jl  (280 chars)
  [FunSQL.jl] Guide  (49284 chars)
  ...
  [web-search] OMOP CDM v5.4  (164111 chars)

Answer:
From(:condition_occurrence) |> Group(Get.person_id) |> Select(...)
```

## Notes and tips

- **Resilience.** Curated entries may list several candidate URLs; unreachable
  ones are skipped, so a single dead link never fails the whole run.
- **Provenance.** Index chunks record their origin URL, so retrieved context is
  traceable back to the source page.
- **Rate limits.** The keyless [`DuckDuckGoProvider`](@ref) is rate-limited and
  best-effort. For heavy use, add a keyed provider via the
  [`AbstractSearchProvider`](@ref) interface.
- **Offline pieces.** [`html_to_text`](@ref) needs no network and is handy to
  test in isolation.

See the [API reference](index.md) for full docstrings of every function above.
```
