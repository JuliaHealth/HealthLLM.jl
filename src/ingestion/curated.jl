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
