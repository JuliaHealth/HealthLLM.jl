"""
    Prompt

Prompt construction for retrieval-augmented FunSQL generation. This module sits
between retrieval and the language model: it takes the chunks a vector store
returned for a question — OMOP schema definitions, FunSQL examples, prose docs —
and the user's natural-language *analytical* question, and renders one grounded
prompt asking the model to translate the question into a FunSQL query.

The two moving parts injected into every prompt are:

1. **Retrieved context** — the ranked chunks from [`retrieve`](@ref HealthLLM.Storage.retrieve)/[`search`](@ref HealthLLM.Storage.search)
   (or raw strings / [`Chunk`](@ref HealthLLM.Ingestion.Chunk)s), formatted as numbered, optionally
   provenance-tagged blocks so the model can ground its answer and cite sources.
2. **The analytical question** — the user's natural-language request, verbatim.

## Interface

- [`PromptTemplate`](@ref) — the reusable shape (system message, section headers,
  per-chunk formatting, and context budgets). [`DEFAULT_FUNSQL_TEMPLATE`](@ref) is
  the OMOP/FunSQL default.
- [`format_context`](@ref)`(template, hits)` — render retrieved chunks to a string.
- [`build_prompt`](@ref)`(question, hits; template)` — the entry point; returns
  `(; system, user, prompt)`.

## Example

```julia
store = LocalVectorStore(embedding_dimension())
add!(store, embed(chunks), chunks)

hits = retrieve(store, "How many distinct patients had a diabetes diagnosis in 2020?", 6)
p = build_prompt("How many distinct patients had a diabetes diagnosis in 2020?", hits)

# split form, for a proper system/user conversation:
msg = PromptingTools.aigenerate(p.system * "\\n\\n" * p.user; model="llama3.2")
# or the single joined string, for the {input_query}-style path:
p.prompt
```
"""
module Prompt

export FUNSQL_SYSTEM_PROMPT, PromptTemplate, DEFAULT_FUNSQL_TEMPLATE,
    format_context, build_prompt

"""
    FUNSQL_SYSTEM_PROMPT

Default system message for FunSQL generation over the OMOP CDM. It fixes the
model's role and the rules that keep generations grounded: build queries only
from tables/columns that appear in the retrieved context, follow the retrieved
examples' FunSQL style, flag missing schema instead of inventing it, and return
the query as a single fenced `julia` code block.

The **Grounding** rules are the load-bearing part: they instruct the model to
treat the retrieved context as the *only* admissible source of schema
(table/column) names and FunSQL syntax, and to refuse rather than hallucinate
when the context is insufficient. Weaken them and generations drift toward
plausible-but-nonexistent OMOP columns.
"""
const FUNSQL_SYSTEM_PROMPT = """
You are an expert data analyst for the OHDSI OMOP Common Data Model (CDM), \
writing analytical queries with the Julia package FunSQL.jl.

Your task: translate the user's natural-language analytical question into a \
correct FunSQL query against the OMOP CDM, using ONLY the retrieved context below.

Grounding (strict — this overrides your prior knowledge):
- The retrieved context is your ONLY source of truth for table names, column \
  names, and FunSQL syntax. Every table and column you reference MUST appear \
  verbatim in the context. Do not rely on memory of the OMOP CDM or of FunSQL, \
  and do not invent, guess, pluralise, or "correct" any identifier.
- Before using a table or column, confirm it is present in the context. If a name \
  you need is not there, DO NOT substitute a similar-looking one — treat it as \
  unavailable.
- Reproduce FunSQL constructs (`From`, `Where`, `Join`, `Group`, `Select`, \
  `Get`, `Agg`, `Fun`, `|>`) only as they appear in the retrieved examples. Do \
  not use SQL string syntax or FunSQL features you cannot see demonstrated in the \
  context.
- If the context lacks a table, column, or construct required to answer the \
  question, state exactly what is missing, then generate the closest correct \
  query you can from what IS available, labelling any assumption. Never paper over \
  a gap with a fabricated identifier.

Style:
- Follow the idioms of the retrieved FunSQL examples.
- Join through the standard OMOP keys shown in the context (e.g. `person_id`) and \
  filter on the `*_concept_id` columns present in the context where the question \
  implies a clinical concept.

Output:
- Return the query as ONE fenced code block tagged `julia`, followed by a one- or \
  two-sentence explanation that names the context tables/columns you used. Do not \
  pad the answer with unrelated commentary.
"""

"""
    PromptTemplate(; kwargs...)

Reusable shape for a retrieval-augmented prompt. A template is data — the
`system` message, the section headers, how each retrieved chunk is rendered, and
how much context to admit — so the same construction logic serves different
models and tasks by swapping fields.

# Fields
- `system::String`: System message. Defaults to [`FUNSQL_SYSTEM_PROMPT`](@ref).
- `context_header::String`: Heading printed above the retrieved chunks.
- `question_header::String`: Heading printed above the user's question.
- `answer_cue::String`: Trailing line that cues the model to answer (e.g. a
  `# FunSQL query` header). Empty to omit.
- `empty_context_note::String`: Placeholder used when no chunks are supplied, so
  the model is told the context is empty rather than seeing a blank section.
- `chunk_label::String`: Per-chunk label prefix; numbered as `"[<label> i]"`.
- `include_provenance::Bool`: Append each chunk's provenance (url/source ›
  heading) when available.
- `include_scores::Bool`: Append each chunk's retrieval score when available.
- `max_chunks::Int`: Keep at most this many chunks (highest-ranked first).
- `max_context_chars::Int`: Stop admitting chunks once the rendered context would
  exceed this many characters. Whole chunks only — a chunk is admitted or skipped,
  never cut mid-text — except a single chunk longer than the budget on its own,
  which is truncated with an ellipsis.
- `separator::String`: Placed between rendered chunks.
"""
Base.@kwdef struct PromptTemplate
    system::String = FUNSQL_SYSTEM_PROMPT
    context_header::String = "# Retrieved context"
    question_header::String = "# Analytical question"
    answer_cue::String = "# FunSQL query"
    empty_context_note::String = "(no relevant context was retrieved)"
    chunk_label::String = "Source"
    include_provenance::Bool = true
    include_scores::Bool = false
    max_chunks::Int = 8
    max_context_chars::Int = 8000
    separator::String = "\n\n"
end

"""
    DEFAULT_FUNSQL_TEMPLATE

The default [`PromptTemplate`](@ref): FunSQL/OMOP system message, provenance on,
scores off, up to 8 chunks or 8000 characters of context.
"""
const DEFAULT_FUNSQL_TEMPLATE = PromptTemplate()

# ---------------------------------------------------------------------------
# Context-item normalisation
# ---------------------------------------------------------------------------

# Duck-typed so `Prompt` need not depend on `Storage` or `Ingestion`. Accepts:
#   * a plain `String`                      -> text only
#   * a search/retrieve hit NamedTuple      -> `.chunk` text, `.score`/`.distance`
#   * a `Chunk` (from Ingestion)            -> `.text`, `.metadata` provenance
# Returns `(; text, provenance, score)` with empty/nothing where unavailable.
function _as_context_item(x)
    x isa AbstractString && return (; text=String(x), provenance="", score=nothing)

    text = hasproperty(x, :chunk) ? getproperty(x, :chunk) :
           hasproperty(x, :text) ? getproperty(x, :text) :
           string(x)

    score = hasproperty(x, :score) ? getproperty(x, :score) :
            hasproperty(x, :distance) ? getproperty(x, :distance) :
            nothing

    provenance = ""
    if hasproperty(x, :metadata) && getproperty(x, :metadata) isa AbstractDict
        provenance = _provenance_from_metadata(getproperty(x, :metadata))
    end

    return (; text=String(text), provenance=provenance, score=score)
end

# Mirror of Ingestion.chunk_provenance, kept local so Prompt stays decoupled:
# "<url-or-source> › <heading-or-group>".
function _provenance_from_metadata(md::AbstractDict)
    base = get(md, :url, "")
    isempty(base) && (base = get(md, :source, ""))
    parent = get(md, :heading, get(md, :group, ""))
    base = string(base)
    parent = string(parent)
    isempty(parent) && return base
    isempty(base) && return parent
    return string(base, " › ", parent)
end

_score_str(::Nothing) = ""
_score_str(s::Real) = string(" (score ", round(float(s), digits=3), ")")
_score_str(s) = ""

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

"""
    format_context(template::PromptTemplate, hits) -> String

Render the retrieved `hits` into a numbered context block per `template`. `hits`
is any iterable of retrieval results — plain strings, `search`/`retrieve` hit
NamedTuples, or [`Chunk`](@ref HealthLLM.Ingestion.Chunk)s. Blank chunks are dropped; the first
`template.max_chunks` are kept, and chunks are admitted only while the running
size stays within `template.max_context_chars` (a single over-budget chunk is
truncated). Returns `template.empty_context_note` when nothing survives.
"""
function format_context(template::PromptTemplate, hits)
    blocks = String[]
    used = 0
    kept = 0
    for h in hits
        kept >= template.max_chunks && break
        item = _as_context_item(h)
        body = strip(item.text)
        isempty(body) && continue

        remaining = template.max_context_chars - used
        remaining <= 0 && break
        if length(body) > remaining
            # Only truncate when this is the first chunk and it alone exceeds the
            # budget; otherwise stop cleanly at a chunk boundary.
            kept == 0 || break
            body = string(first(body, max(remaining - 1, 0)), "…")
        end

        kept += 1
        header = string("[", template.chunk_label, " ", kept, "]")
        if template.include_scores
            header *= _score_str(item.score)
        end
        if template.include_provenance && !isempty(item.provenance)
            header *= string("  ", item.provenance)
        end
        block = string(header, "\n", body)
        push!(blocks, block)
        used += length(block) + length(template.separator)
    end

    isempty(blocks) && return template.empty_context_note
    return join(blocks, template.separator)
end

"""
    build_prompt(question, hits; template=DEFAULT_FUNSQL_TEMPLATE)
        -> (; system, user, prompt)

Construct a retrieval-augmented prompt from the user's natural-language
analytical `question` and the retrieved `hits`, using `template`.

Returns a NamedTuple with three fields, so both prompting styles are covered:
- `system::String` — the template's system message, for a proper system/user
  conversation (e.g. `PromptingTools.aigenerate` with a system role).
- `user::String` — the user turn: the formatted context block, the question, and
  the answer cue.
- `prompt::String` — `system` and `user` joined, for single-string callers.

`hits` accepts plain strings, `search`/`retrieve` hit NamedTuples, or
[`Chunk`](@ref HealthLLM.Ingestion.Chunk)s (see [`format_context`](@ref)). Throws `ArgumentError` on an
empty `question`.

# Example

```julia
hits = retrieve(store, "Average age of patients with hypertension?", 6)
p = build_prompt("Average age of patients with hypertension?", hits)
println(p.prompt)
```
"""
function build_prompt(question::AbstractString, hits;
    template::PromptTemplate=DEFAULT_FUNSQL_TEMPLATE)
    q = strip(question)
    isempty(q) && throw(ArgumentError("`question` is empty; nothing to ask."))

    context = format_context(template, hits)

    parts = String[]
    push!(parts, template.context_header, context, template.question_header, String(q))
    isempty(template.answer_cue) || push!(parts, template.answer_cue)
    user = join(parts, "\n\n")

    prompt = string(template.system, "\n\n", user)
    return (; system=template.system, user=user, prompt=prompt)
end

"""
    build_prompt(question; template=DEFAULT_FUNSQL_TEMPLATE) -> (; system, user, prompt)

Convenience method for a question with no retrieved context — renders the
template's `empty_context_note` in the context slot.
"""
build_prompt(question::AbstractString; template::PromptTemplate=DEFAULT_FUNSQL_TEMPLATE) =
    build_prompt(question, (); template=template)

end
