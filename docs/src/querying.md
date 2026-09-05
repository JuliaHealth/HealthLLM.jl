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

## From RAG response to a database query

The model's reply is prose with a FunSQL code block embedded in it — not something
you can run directly. Turning that reply into an executed query is three moves:
**generate**, **extract**, **render to SQL**, then hand the SQL to a database.

```
grounded prompt ─▶ generate_funsql ─▶ .funsql (code) ─▶ eval ─▶ SQLNode ─▶ FunSQL.render ─▶ SQL ─▶ DB
```

[`generate_funsql`](@ref) sends the prompt to the chat model and pulls the FunSQL
code out of the reply for you (via [`extract_funsql`](@ref), which prefers the
first ` ```julia ` block):

```julia
gen = generate_funsql(p; model = "llama3.2")
gen.funsql      # just the FunSQL code, e.g. "From(:person) |> Group() |> Select(...)"
gen.answer      # the full reply, including the model's explanation
```

FunSQL is a **query builder**: the code evaluates to a `SQLNode`, which
`FunSQL.render` serialises to SQL for a specific dialect. You then execute that SQL
with whatever driver your data lives behind — [DuckDB](https://duckdb.org) for the
JuliaHealth demo datasets, or `LibPQ` for PostgreSQL. FunSQL and the driver are
**driver-side dependencies** (not pulled in by HealthLLM), so load them yourself:

```julia
using FunSQL, DuckDB, DataFrames

conn = DuckDB.DB("synthea.duckdb")

# 1. Evaluate the generated code into a FunSQL query node.
node = eval(Meta.parse(gen.funsql))

# 2. Render it to DuckDB SQL.
sql = FunSQL.render(node; dialect = :duckdb)

# 3. Execute against the database.
result = DuckDB.execute(conn, String(sql)) |> DataFrame
```

!!! warning "You are evaluating model output"
    `eval`-ing generated code runs it with full privileges. Only do this with a
    model you trust, sandbox the database it can reach, and never point the
    execution path at production data. This is why the [sanity
    check](#Sanity-checking-the-generated-query) below renders (and optionally
    runs) against a *disposable* schema/connection first.

Rendering against a **schema catalog** rather than a bare dialect is what makes the
query trustworthy — FunSQL resolves every table and column against the catalog and
errors on anything it does not find. Build a catalog by hand, or reflect one from a
live connection:

```julia
catalog = FunSQL.reflect(conn; dialect = :duckdb)   # discovers the DB's tables
sql = FunSQL.render(catalog, node)                   # fails on unknown table/column
```

## Sanity-checking the generated query

Before running generated SQL — or surfacing it to a user — check that it is
actually valid. [`sanity_check_funsql`](@ref) runs the query as far as it can
through a **parse → build → render → run** pipeline and reports the outcome as a
[`FunSQLCheck`](@ref), without ever throwing on a bad query:

| Stage      | What it proves                                      | Needs                     |
|------------|-----------------------------------------------------|---------------------------|
| `parsed`   | the code is syntactically valid Julia               | nothing                   |
| `built`    | it evaluates to a `FunSQL.SQLNode`                  | `using FunSQL`            |
| `rendered` | it serialises to SQL — **resolved against a schema**| a `catalog` (recommended) |
| `executed` | the database accepts the SQL                        | an `executor`             |

```julia
using FunSQL

catalog = FunSQL.SQLCatalog(
    FunSQL.SQLTable(:person, columns = [:person_id, :year_of_birth]),
    FunSQL.SQLTable(:condition_occurrence,
        columns = [:condition_occurrence_id, :person_id, :condition_concept_id]),
    dialect = :duckdb,
)

check = sanity_check_funsql(gen; catalog = catalog)
check.ok       # true if it parsed, built, and rendered
check.sql      # the rendered SQL, when rendering succeeded
```

The `catalog` stage is the **execution-based grounding check**: it catches a
hallucinated OMOP name that slipped past the prompt's grounding rules, because
FunSQL refuses to resolve a column the schema does not contain.

```julia
# The model invented a column that isn't in the schema:
bad = sanity_check_funsql("From(:person) |> Select(Get.made_up_column)"; catalog = catalog)
bad.ok         # false
bad.stage      # :built  — it built, but failed to resolve against the schema
bad.error      # message naming the unresolved reference
```

Pass an `executor` to go one stage further and confirm the database itself accepts
the SQL — the check stays failure-safe, capturing any driver error instead of
throwing:

```julia
using DuckDB
conn = DuckDB.DB("synthea.duckdb")

check = sanity_check_funsql(gen; catalog = catalog,
                            executor = sql -> DuckDB.execute(conn, sql))
check.executed        # true if the query ran
```

A natural loop is: generate → sanity-check → if `!check.ok`, feed `check.error`
back to the model (alongside the same context) and regenerate, so a hallucinated
name becomes a corrective signal rather than a silent failure.

## End to end

```julia
using HealthLLM, PromptingTools, FunSQL, DuckDB, DataFrames

register_models("llama3.2", "nomic-embed-text")

# 1. Load (or build) the index and open the database.
store = load(LocalVectorStore, "omop_index.jls")
conn  = DuckDB.DB("synthea.duckdb")
catalog = FunSQL.reflect(conn; dialect = :duckdb)     # schema to validate against

# 2. Retrieve context for the question and build the grounded prompt.
question = "What is the average age of patients with a hypertension diagnosis?"
hits     = retrieve(store, question, 6)
p        = build_prompt(question, hits)

# 3. Generate the FunSQL query from the grounded prompt.
gen = generate_funsql(p; model = "llama3.2")

# 4. Sanity-check it against the schema before trusting it.
check = sanity_check_funsql(gen; catalog = catalog)
if !check.ok
    error("Generated query failed at :$(check.stage) — $(check.error)")
end

# 5. Execute the validated SQL.
result = DuckDB.execute(conn, check.sql) |> DataFrame
```

See the [API reference](index.md) for full docstrings of [`build_prompt`](@ref),
[`generate_funsql`](@ref), [`sanity_check_funsql`](@ref), [`FunSQLCheck`](@ref),
and [`FUNSQL_SYSTEM_PROMPT`](@ref).
```
