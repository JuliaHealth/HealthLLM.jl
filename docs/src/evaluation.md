# Evaluation

```@meta
CurrentModule = HealthLLM.Evaluation
```

The `Evaluation` module measures whether retrieval grounding improves
natural-language → FunSQL generation. Each benchmark task is answered by the same
model twice:

- **zero-shot**: the question alone, with a system prompt that asks for FunSQL
  over the OMOP CDM;
- **RAG**: the question plus chunks retrieved from the grounding docs, with
  [`FUNSQL_SYSTEM_PROMPT`](@ref HealthLLM.Prompt.FUNSQL_SYSTEM_PROMPT).

Each generated query is scored by **running it**. It must parse, build a FunSQL
query, render against the real database schema, execute, and return the same
result as the task's gold SQL. String similarity to a reference answer is never
used. The two runs are then compared task by task.

## Running the benchmark

You need:

- an OMOP CDM DuckDB file. The benchmark was built on `synthea_1M_3YR.duckdb`, a
  Synthea ETL with its tables in schema `dbt_synthea_dev`;
- [Ollama](https://ollama.com) serving a chat model and `nomic-embed-text`.

```bash
julia --project=scripts/benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts/benchmark scripts/benchmark/run_benchmark.jl \
    --db synthea_1M_3YR.duckdb --model qwen2.5:7b
```

On Julia 1.10, also run `Pkg.develop(path=".")` in that environment, because
the `[sources]` entry pointing at the repository needs Julia 1.11 or later.

The script writes `benchmark_results/<timestamp>/results.json`, which holds every
task's prompt context, generated code, rendered SQL, and outcome. It also writes
`report.md`, a summary you can paste into a write-up. Pass `--limit 5` for a
quick check, `--tiers schema_aware` to run one tier, or `--modes rag` to skip the
baseline. The options are listed at the top of the script.

Generated code is evaluated in Julia to build the query, as
[`sanity_check_funsql`](@ref HealthLLM.Execution.sanity_check_funsql) documents.
Run the benchmark only with models you trust, on a machine you don't mind them
running code on. The database is attached **read-only**. The gold SQL's `cdm.<table>` names are
views over it, so no generated query can modify the source file. Every query runs
under a timeout, which stops a runaway join over the full patient table.

## The task set

`data/benchmark/funsql_tasks.jsonl` holds 180 tasks. Each has a self-contained
question, DuckDB gold SQL, and a FunSQL answer verified to return the gold SQL's
result on the Synthea 1M database. A task is kept only if its gold SQL returns at
least one row, because an empty result would be matched by any query that also
returns nothing.

The tasks are a repair of the 200-row FunSQL example dataset
(`FunSQLQueries/train.jsonl`, adapted from the OHDSI QueryLibrary).
`data/benchmark/README.md` describes how each record was rebuilt and why 20 were
dropped. In short: the original SQL was mostly SQL Server dialect, many questions
didn't match their SQL or left out the concept ids it used, and most input
concepts never occur in Synthea. The editable source is
`data/benchmark/funsql_duckdb_source.jsonl`. `scripts/benchmark/curate_tasks.jl`
re-verifies it to produce the benchmark file, and
`scripts/benchmark/verify_pairs.jl` checks individual SQL/FunSQL pairs while
editing.

Each task has a tier, assigned by [`classify_tier`](@ref) from its gold SQL:

| Tier | Tasks | Needs |
|------|------:|-------|
| `simple` | 45 | one table: filter, group, aggregate, order |
| `compositional` | 43 | joins, subqueries, CTEs, window or set operations |
| `schema_aware` | 92 | OMOP vocabulary tables (`concept`, `concept_ancestor`, `concept_relationship`) |

## Scoring

Each [`TaskResult`](@ref) records how far the generated query got:

| Outcome | Meaning |
|---------|---------|
| `correct` | Executed and matched the gold result |
| `mismatch` | Executed and returned a different result |
| `execute_error` | Rendered, but the database rejected the SQL |
| `render_error` | Built, but referenced a table or column the schema lacks |
| `build_error` | Valid Julia that is not a FunSQL query |
| `parse_error` | Not valid Julia (e.g. SQL, or SQL-like pseudocode) |
| `generation_error` | The model call failed |
| `gold_error` | The task's own gold SQL failed. Excluded from all rates |

[`results_match`](@ref) defines "the same result". It compares columns by content
rather than alias, ignores row order, and compares numbers with a tolerance.
[`compare_modes`](@ref) pairs the two modes per task and reports the discordant
counts with an exact McNemar test, so you can tell a RAG gain from noise.

### Leakage

If a task's own reference answer is in the index, retrieving it measures lookup,
not grounding. [`split_tasks`](@ref) holds tasks out with a deterministic,
tier-stratified split. Only the *other* tasks' verified FunSQL is indexed, via
[`examples_document`](@ref). The default `--test-fraction 0.5` evaluates half
the tasks and indexes the other half's verified FunSQL as retrievable examples.
`--test-fraction 1.0` evaluates every task and grounds the RAG mode on
`data/grounding/` alone.

## Using the harness from Julia

The harness reaches the database, the model, and the retriever only through
functions you pass in, so it can be driven from Julia directly or tested offline:

```julia
using HealthLLM, FunSQL

tasks = load_tasks(joinpath(pkgdir(HealthLLM), "data", "benchmark", "funsql_tasks.jsonl"))

results = run_benchmark(tasks;
    model     = "qwen2.5:7b",
    executor  = sql -> my_resultset(sql),   # SQL -> ResultSet
    catalog   = my_funsql_catalog,          # FunSQL.SQLCatalog of the database
    retriever = q -> retrieve(store, q, 6)) # question -> hits

summarize(results)[:rag].overall.execution_accuracy
compare_modes(results).p_value
print(markdown_report(results))
```

`scripts/benchmark/omop_duckdb.jl` provides `open_omop_duckdb` and
`duckdb_executor` for DuckDB.
