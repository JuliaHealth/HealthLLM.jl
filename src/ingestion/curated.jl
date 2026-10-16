"""
    CURATED_SOURCES

Registry of curated documentation sources as `name => Vector{url}`.

Each entry lists one or more URLs for a documentation source. Raw Markdown
endpoints (e.g. `raw.githubusercontent.com`) are preferred because they need no
HTML cleaning; rendered pages are also accepted and stripped to text on fetch.

URLs that fail to fetch (404, timeout, network error) are skipped, so entries
may safely list several candidate locations for robustness. Extend or override
this registry to point at your own curated set.
"""
const CURATED_SOURCES = Dict{String,Vector{String}}(
    "JuliaHealth" => [
        "https://raw.githubusercontent.com/JuliaHealth/juliahealth.github.io/main/JuliaHealthBlog/posts/juliahealth-ecosystem/ecosystem.qmd",
        "https://juliahealth.org/",
    ],
    "OMOP CDM" => [
        "https://raw.githubusercontent.com/OHDSI/CommonDataModel/main/README.md",
        "https://ohdsi.github.io/CommonDataModel/cdm54.html",
    ],
    "OHDSI" => [
        "https://raw.githubusercontent.com/OHDSI/TheBookOfOhdsi/master/StandardizedVocabularies.Rmd",
        "https://www.ohdsi.org/data-standardization/",
    ],
    "FunSQL.jl" => [
        "https://raw.githubusercontent.com/MechanicalRabbit/FunSQL.jl/master/docs/src/index.md",
        "https://raw.githubusercontent.com/MechanicalRabbit/FunSQL.jl/master/docs/src/guide/index.md",
        "https://raw.githubusercontent.com/MechanicalRabbit/FunSQL.jl/master/docs/src/examples/index.md",
    ],
)

"""
    GROUNDING_DIR

Directory of the local grounding documents shipped with HealthLLM
(`data/grounding/`): an OMOP CDM schema reference, a FunSQL.jl reference, and
SQL → FunSQL translation patterns. Every FunSQL snippet in them is checked to
render against the documented schema by the test suite, so they are safe to use
as the model's only source of syntax. Load them with [`load_grounding_docs`](@ref).
"""
const GROUNDING_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "grounding"))

"""
    load_grounding_docs(dir=GROUNDING_DIR) -> Vector{SourceDocument}

Load every Markdown file in `dir` as a [`SourceDocument`](@ref), with source
`"grounding:<file name>"` and the file's first heading as its title. They are
Markdown with one heading per table or pattern, so [`default_strategy`](@ref)
chunks them with [`HeaderChunk`](@ref) and each OMOP table definition or
translation pattern becomes one retrievable chunk.

Unlike [`fetch_curated`](@ref) this needs no network, which makes it the
reproducible grounding source for benchmarks.

# Example

```julia
examples = joinpath(pkgdir(HealthLLM), "data", "benchmark", "funsql_tasks.jsonl")
docs = vcat(load_grounding_docs(), load_funsql_examples(examples))
index = ingest_to_index(; docs=docs, embedder_kwargs=(model="nomic-embed-text",))
```
"""
function load_grounding_docs(dir::AbstractString=GROUNDING_DIR)
    isdir(dir) || throw(ArgumentError("No such grounding directory: $dir"))
    docs = SourceDocument[]
    for name in sort(readdir(dir))
        endswith(lowercase(name), ".md") || continue
        path = joinpath(dir, name)
        content = read(path, String)
        isempty(strip(content)) && continue
        push!(docs, SourceDocument("grounding:" * splitext(name)[1], path,
            _title_from(content, path), content))
    end
    return docs
end
