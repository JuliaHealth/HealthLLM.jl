using PromptingTools: recursive_splitter
using JSON3

"""
    Chunk

One retrieval chunk plus grounding metadata.

# Fields
- `text::String`: The chunk text.
- `metadata::Dict{Symbol,Any}`: Grounding info such as `:heading` (the parent
  Markdown/section heading, e.g. the OMOP table a chunk describes), `:group`
  (FunSQL example group), `:source`, `:url`, and `:title`. Populated by the
  strategy and enriched with the document's provenance by [`chunk_document`](@ref).
"""
struct Chunk
    text::String
    metadata::Dict{Symbol,Any}
end
Chunk(text::AbstractString) = Chunk(String(text), Dict{Symbol,Any}())

"""
    AbstractChunkStrategy

Supertype for chunking strategies. A strategy turns one document's text into a
vector of [`Chunk`](@ref)s via [`chunk`](@ref)`(strategy, text)`.

Concrete strategies:
- [`RecursiveChunk`](@ref)   — size-bounded recursive split (default for prose).
- [`HeaderChunk`](@ref)      — one chunk per Markdown heading section (OMOP tables).
- [`RecordChunk`](@ref)      — one chunk per JSONL record (FunSQL examples).
- [`FixedSizeChunk`](@ref)   — fixed character windows with overlap.
"""
abstract type AbstractChunkStrategy end

"""
    chunk(strategy::AbstractChunkStrategy, text::AbstractString) -> Vector{Chunk}

Split `text` into atomic [`Chunk`](@ref)s according to `strategy`. Fenced code
blocks are never split mid-expression, and each chunk carries the parent heading
it came from where the strategy can determine one. Empty/whitespace-only chunks
are dropped.
"""
function chunk end

const _RECURSIVE_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

"""
    RecursiveChunk(; max_length=1024, separators=_RECURSIVE_SEPARATORS)

Split text into heading sections, then split each section to at or below
`max_length` characters, preferring the earliest separator that fits. Fenced code
blocks (```` ``` ````/`~~~`) are kept whole regardless of length so a FunSQL
snippet is never cut mid-expression. Each resulting chunk records its parent
`:heading`. The right default for prose docs (FunSQL guide, OHDSI book,
JuliaHealth), where a size cut degrades gracefully.
"""
Base.@kwdef struct RecursiveChunk <: AbstractChunkStrategy
    max_length::Int = 1024
    separators::Vector{String} = _RECURSIVE_SEPARATORS
end

function chunk(s::RecursiveChunk, text::AbstractString)
    out = Chunk[]
    for (heading, body) in _sections(String(text))
        for piece in _split_preserving_code(body, s.separators, s.max_length)
            _push_chunk!(out, piece, heading)
        end
    end
    return out
end

"""
    HeaderChunk(; min_level=1, max_level=6, max_length=4096)

Split Markdown on ATX heading boundaries (`#`..`######`), emitting one chunk per
heading together with the body beneath it, up to the next heading of the same or
higher level. This keeps an OMOP table definition (its name heading plus the full
column/foreign-key list) in a single chunk, tagged with `:heading`.

A section longer than `max_length` characters is further split with the same
code-preserving splitter as [`RecursiveChunk`](@ref), so no single chunk blows
past embedding limits and no code block is broken. Text before the first heading
becomes its own leading chunk. Heading lines inside fenced code blocks are
ignored. `min_level`/`max_level` bound which heading depths start a new section
(e.g. `min_level=2` ignores the document title `#` and breaks on `##` headings).
"""
Base.@kwdef struct HeaderChunk <: AbstractChunkStrategy
    min_level::Int = 1
    max_level::Int = 6
    max_length::Int = 4096
end

function chunk(s::HeaderChunk, text::AbstractString)
    out = Chunk[]
    for (heading, body) in _sections(String(text); min_level=s.min_level, max_level=s.max_level)
        pieces = length(body) > s.max_length ?
                 _split_preserving_code(body, _RECURSIVE_SEPARATORS, s.max_length) : [body]
        for piece in pieces
            _push_chunk!(out, piece, heading)
        end
    end
    return out
end

"""
    RecordChunk(; fields=["query", "description", "sql_query", "response"],
                  labels=["Query", "Description", "SQL", "FunSQL"])

Treat the input as JSON Lines (one JSON object per line) and emit exactly one
chunk per record — the atomic retrieval unit for the FunSQL example dataset in
`FunSQLQueries/`.

`fields` are pulled from each record in order and rendered as
`\"<label>: <value>\"` lines. The natural-language `query`/`description` lead so
the embedding emphasises the question side, while the `sql_query`/`response`
travel in the same chunk as the retrievable answer. Each chunk keeps the record's
`group` and `query` in `:group`/`:heading` metadata for grounding. Missing fields
are skipped; lines that fail to parse as JSON are ignored.
"""
Base.@kwdef struct RecordChunk <: AbstractChunkStrategy
    fields::Vector{String} = ["query", "description", "sql_query", "response"]
    labels::Vector{String} = ["Query", "Description", "SQL", "FunSQL"]
end

function chunk(s::RecordChunk, text::AbstractString)
    @assert length(s.fields) == length(s.labels) "`fields` and `labels` must be the same length"
    out = Chunk[]
    for line in split(String(text), '\n')
        isempty(strip(line)) && continue
        rec = try
            JSON3.read(line)
        catch
            @debug "Skipping unparseable JSONL line while chunking records"
            continue
        end
        parts = String[]
        for (f, label) in zip(s.fields, s.labels)
            key = Symbol(f)
            if haskey(rec, key)
                val = strip(string(rec[key]))
                isempty(val) || push!(parts, "$label: $val")
            end
        end
        isempty(parts) && continue
        meta = Dict{Symbol,Any}()
        haskey(rec, :group) && (meta[:group] = strip(string(rec[:group])))
        haskey(rec, :query) && (meta[:heading] = strip(string(rec[:query])))
        push!(out, Chunk(join(parts, '\n'), meta))
    end
    return out
end

"""
    FixedSizeChunk(; size=512, overlap=64)

Split into fixed windows of `size` characters that overlap their neighbours by
`overlap` characters. Character-based and structure-blind (it will split code
mid-expression) — provided as a baseline for comparison; prefer
[`RecursiveChunk`](@ref) for real prose.
"""
Base.@kwdef struct FixedSizeChunk <: AbstractChunkStrategy
    size::Int = 512
    overlap::Int = 64
end

function chunk(s::FixedSizeChunk, text::AbstractString)
    @assert s.size > s.overlap >= 0 "require size > overlap >= 0"
    cs = collect(String(text))
    n = length(cs)
    n == 0 && return Chunk[]
    step = s.size - s.overlap
    out = Chunk[]
    i = 1
    while i <= n
        _push_chunk!(out, String(cs[i:min(i + s.size - 1, n)]), "")
        i += step
    end
    return out
end

"""
    default_strategy(doc::SourceDocument) -> AbstractChunkStrategy

Pick a chunking strategy from a document's source/content:

| Detected content                | Strategy                          |
|----------------------------------|-----------------------------------|
| FunSQL example JSONL             | [`RecordChunk`](@ref)             |
| OMOP / Markdown-heading docs     | [`HeaderChunk`](@ref)             |
| everything else (prose)          | [`RecursiveChunk`](@ref)          |

Detection is by source name first (e.g. the `"FunSQL-examples"` source produced
by [`load_funsql_examples`](@ref)), then by a content sniff so ad-hoc documents
are still routed sensibly.
"""
function default_strategy(doc::SourceDocument)
    src = lowercase(doc.source)
    occursin("funsql-example", src) && return RecordChunk()
    _looks_like_jsonl(doc.content) && return RecordChunk()
    (occursin("omop", src) || occursin("cdm", src) || _looks_like_markdown_headers(doc.content)) &&
        return HeaderChunk()
    return RecursiveChunk()
end

"""
    chunk_document(doc::SourceDocument; strategy=default_strategy(doc)) -> Vector{Chunk}

Chunk one [`SourceDocument`](@ref) at the granularity appropriate to its content
type (or an explicit `strategy`), enriching each chunk's metadata with the
document's `:source`, `:url`, and `:title` for later grounding.
"""
function chunk_document(doc::SourceDocument; strategy::AbstractChunkStrategy=default_strategy(doc))
    chunks = chunk(strategy, doc.content)
    for c in chunks
        c.metadata[:source] = doc.source
        isempty(doc.url) || (c.metadata[:url] = doc.url)
        isempty(doc.title) || get!(c.metadata, :title, doc.title)
    end
    return chunks
end

"""
    chunk_provenance(c::Chunk) -> String

Render a chunk's grounding as a single provenance string (`"<url-or-source> ›
<parent>"`, truncated to 512 chars) suitable for RAGTools chunk sources.

Delegates to [`render_provenance`](@ref), which is also what the prompt layer uses
to tag a retrieved chunk — so what is stored alongside a vector and what the model
is shown are the same string.
"""
chunk_provenance(c::Chunk) = render_provenance(c.metadata)

"""
    load_funsql_examples(path; source="FunSQL-examples") -> Vector{SourceDocument}

Load a local JSON Lines file of FunSQL examples (see `FunSQLQueries/train.jsonl`)
into a single [`SourceDocument`](@ref) whose `content` is the raw JSONL. Routed
through [`RecordChunk`](@ref) at index time, it yields one chunk per example.
The file path is recorded as provenance.
"""
function load_funsql_examples(path::AbstractString; source::AbstractString="FunSQL-examples")
    isfile(path) || throw(ArgumentError("No such FunSQL example file: $path"))
    content = read(path, String)
    return [SourceDocument(source, path, basename(path), content)]
end

function _push_chunk!(out::Vector{Chunk}, text::AbstractString, heading::AbstractString)
    t = strip(text)
    isempty(t) && return out
    meta = Dict{Symbol,Any}()
    isempty(heading) || (meta[:heading] = String(heading))
    push!(out, Chunk(String(t), meta))
    return out
end

const _FENCE = r"^\s*(```|~~~)"

function _sections(text::AbstractString; min_level::Integer=1, max_level::Integer=6)
    sections = Tuple{String,String}[]
    heading = ""
    current = String[]
    infence = false
    flush!() = isempty(current) || push!(sections, (heading, join(current, '\n')))
    for line in split(String(text), '\n')
        if occursin(_FENCE, line)
            infence = !infence
            push!(current, line)
            continue
        end
        m = infence ? nothing : match(r"^(#{1,6})\s+(\S.*)$", line)
        if m !== nothing && min_level <= length(m[1]) <= max_level
            flush!()
            current = String[]
            heading = strip(m[2])
        end
        push!(current, line)
    end
    flush!()
    return sections
end

function _code_fence_segments(text::AbstractString)
    segs = Tuple{Bool,String}[]
    buf = String[]
    incode = false
    flush!(iscode) = begin
        isempty(buf) || push!(segs, (iscode, join(buf, '\n')))
        buf = String[]
    end
    for line in split(String(text), '\n')
        if occursin(_FENCE, line)
            if incode
                push!(buf, line)
                flush!(true)
                incode = false
            else
                flush!(false)
                push!(buf, line)
                incode = true
            end
        else
            push!(buf, line)
        end
    end
    flush!(incode)
    return segs
end

function _split_preserving_code(text::AbstractString, separators, max_length::Integer)
    pieces = String[]
    for (iscode, seg) in _code_fence_segments(text)
        if iscode
            isempty(strip(seg)) || push!(pieces, seg)
        else
            for p in recursive_splitter(seg, separators; max_length)
                isempty(strip(p)) || push!(pieces, String(p))
            end
        end
    end
    merged = String[]
    buf = ""
    for p in pieces
        if isempty(buf)
            buf = p
        elseif length(buf) + length(p) + 1 <= max_length
            buf = string(buf, '\n', p)
        else
            push!(merged, buf)
            buf = p
        end
    end
    isempty(buf) || push!(merged, buf)
    return merged
end

function _looks_like_jsonl(text::AbstractString)
    for line in split(String(text), '\n')
        isempty(strip(line)) && continue
        stripped = strip(line)
        (startswith(stripped, "{") && endswith(stripped, "}")) || return false
        try
            JSON3.read(stripped)
            return true
        catch
            return false
        end
    end
    return false
end

_looks_like_markdown_headers(text::AbstractString) =
    count(!isnothing, (match(r"^#{1,6}\s+\S", l) for l in split(String(text), '\n'))) >= 2
