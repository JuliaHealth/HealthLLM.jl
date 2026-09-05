```@meta
CurrentModule = HealthLLM
```

# Querying the RAG System

Once documents are [ingested](ingestion.md) and [embedded](embeddings.md) into a
vector store, querying is a three-step loop: **retrieve** the chunks relevant to
a question, **construct** a grounded prompt from them, and **generate** a FunSQL
answer with a chat model.

```
question ─▶ retrieve ─▶ ranked chunks ─┐
                                        ├─▶ build_prompt ─▶ (; system, user, prompt) ─▶ chat model ─▶ FunSQL
           (schema + docs + examples) ─┘
```

The middle step — [`build_prompt`](@ref) — is what keeps generations honest: it
injects the retrieved schema/doc chunks alongside the natural-language question
and pairs them with a system message that forbids the model from using any table
or column name it cannot see in the context.

## From question to prompt

Assume you already have a populated store (see [Building Embeddings](embeddings.md)):

```julia
using HealthLLM

store = load(LocalVectorStore, "omop_index.jls")
question = "How many distinct patients had a diabetes diagnosis in 2020?"
```

Retrieve the most relevant chunks, then build the prompt from them:

```julia
hits = retrieve(store, question, 6)          # top-6 nearest chunks
p    = build_prompt(question, hits)          # (; system, user, prompt)
```

[`build_prompt`](@ref) returns a NamedTuple with three fields so it fits either
prompting style:

| Field    | What it is                                            | Use it for                                  |
|----------|-------------------------------------------------------|---------------------------------------------|
| `system` | The FunSQL/OMOP system message ([`FUNSQL_SYSTEM_PROMPT`](@ref)) | a proper system/user chat conversation |
| `user`   | Context block + question + answer cue                 | the user turn of that conversation          |
| `prompt` | `system` and `user` joined into one string            | single-string / `{input_query}`-style calls |

The `user` turn interleaves the retrieved context with the question:

```
# Retrieved context

[Source 1]  https://ohdsi.github.io/CommonDataModel/ › condition_occurrence
# condition_occurrence
condition_occurrence_id, person_id, condition_concept_id, condition_start_date, ...

[Source 2]  FunSQL-examples › patients with a condition
Query: patients with a condition
FunSQL: From(condition_occurrence) |> Group(Get.person_id)

# Analytical question

How many distinct patients had a diabetes diagnosis in 2020?

# FunSQL query
```

## Generating the answer

Feed the prompt to a registered chat model through `PromptingTools`. The split
`system`/`user` form is preferred — it puts the grounding rules in the system
role where models weight them most heavily:

```julia
using PromptingTools

register_models("llama3.2", "nomic-embed-text")   # once per session

msg = aigenerate(
    [PromptingTools.SystemMessage(p.system),
     PromptingTools.UserMessage(p.user)];
    model = "llama3.2",
)
println(msg.content)      # the FunSQL query + a short explanation
```

For a store-free index built with RAGTools, [`generate_funsql_query`](@ref)
retrieves and generates in one call against a RAG `index` (see
[Document Ingestion](ingestion.md)):

```julia
index  = ingest_to_index(; sources = ["FunSQL.jl", "OMOP CDM"],
                           query = "OMOP condition_occurrence columns")
answer = generate_funsql_query(index, "nomic-embed-text", "llama3.2",
                               "Context: {input_query}", question)
```

## Grounding: why the model stays on-schema

The single biggest failure mode for text-to-query over a large schema like OMOP
is **hallucinated identifiers** — the model confidently writes `patient_id` or
`diagnosis_date` because those names are plausible, even though the CDM uses
`person_id` and `condition_start_date`. [`FUNSQL_SYSTEM_PROMPT`](@ref) counters
this with strict grounding rules that make the retrieved context the *only*
admissible source of names and syntax:

- **Names must appear verbatim in the context.** Every table and column the model
  references has to be present in a retrieved chunk. Memory of the OMOP CDM is
  explicitly overridden, and inventing, guessing, or "correcting" an identifier is
  forbidden.
- **FunSQL constructs are copied, not recalled.** The model may use `From`,
  `Where`, `Join`, `Group`, `Select`, `Get`, `Agg`, `Fun`, and `|>` only as the
  retrieved examples demonstrate them — no raw SQL strings, no undemonstrated
  features.
- **Missing schema is surfaced, not faked.** When the context lacks a table or
  column the question needs, the model is told to say what is missing and generate
  the closest correct query from what *is* available — never to paper over the gap
  with a fabricated name.

Because grounding is only as good as the context, retrieval quality matters: if
`retrieve` does not surface the `condition_occurrence` schema chunk, no prompt
rule can make the model use it correctly. Two levers help:

1. **Retrieve enough chunks.** A larger `k` (or a larger `max_chunks`) raises the
   chance the needed table definition is present — at the cost of a longer prompt.
2. **Index schema at table granularity.** The [`HeaderChunk`](@ref) strategy keeps
   each OMOP table's full column list in one chunk, so a single hit carries the
   whole definition rather than a fragment.

## Tuning the prompt

[`build_prompt`](@ref) is driven by a [`PromptTemplate`](@ref) — pass one to
change the system message, the section headers, per-chunk formatting, or the
context budget. [`DEFAULT_FUNSQL_TEMPLATE`](@ref) is the OMOP/FunSQL default
(provenance on, scores off, up to 8 chunks or 8000 characters).

```julia
tmpl = PromptTemplate(
    include_scores = true,     # show each chunk's retrieval score
    max_chunks     = 5,        # keep at most 5 chunks
    max_context_chars = 6000,  # ...and cap the context at 6000 chars
)

p = build_prompt(question, hits; template = tmpl)
```

The two budgets protect the context window: chunks are admitted highest-ranked
first, and admission stops at a **chunk boundary** once either limit is reached —
a chunk is included whole or skipped, never cut mid-text (the sole exception is a
single chunk that exceeds the character budget on its own, which is truncated with
an ellipsis). This means the prompt never silently overflows the model's window,
and lower-ranked chunks are the ones dropped when space runs out.

You can render just the context block — for inspection or a custom prompt layout —
with [`format_context`](@ref):

```julia
println(format_context(DEFAULT_FUNSQL_TEMPLATE, hits))
```

### Flexible chunk inputs

`build_prompt` and `format_context` accept whatever your retrieval step produces,
so the same call works across store backends and index paths:

| Input                          | Where it comes from                          |
|--------------------------------|----------------------------------------------|
| `search`/`retrieve` hit NamedTuples | [`LocalVectorStore`](@ref) / [`FaissVectorStore`](@ref) (`.chunk`, `.score`) and [`PgVectorStore`](@ref) (`.chunk`, `.distance`) |
| plain `String`s                | any ad-hoc list of context snippets          |
| [`Chunk`](@ref)s               | the ingestion layer, carrying `:heading`/`:source`/`:url` provenance |

When a chunk carries provenance (a URL or source plus its parent heading), it is
shown in the `[Source i]` header so the model — and you — can trace each fact back
to its origin.

## End to end

```julia
using HealthLLM, PromptingTools

register_models("llama3.2", "nomic-embed-text")

# 1. Load (or build) the index.
store = load(LocalVectorStore, "omop_index.jls")

# 2. Retrieve context for the question.
question = "What is the average age of patients with a hypertension diagnosis?"
hits     = retrieve(store, question, 6)

# 3. Construct the grounded prompt.
p = build_prompt(question, hits)

# 4. Generate the FunSQL query.
msg = aigenerate(
    [PromptingTools.SystemMessage(p.system),
     PromptingTools.UserMessage(p.user)];
    model = "llama3.2",
)
println(msg.content)
```

See the [API reference](index.md) for full docstrings of [`build_prompt`](@ref),
[`format_context`](@ref), [`PromptTemplate`](@ref), and [`FUNSQL_SYSTEM_PROMPT`](@ref).
```
