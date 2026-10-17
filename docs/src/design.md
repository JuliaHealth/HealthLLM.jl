# Design and Limitations

This page explains why HealthLLM is built the way it is, what it can't do yet,
and where it is meant to be extended.

## Pipeline

```
curated docs ─┐                     ┌─ grounding docs (data/grounding/)
web search ───┼─▶ Ingestion ─▶ chunk ┤
              │   (SourceDocument)   └─ verified FunSQL examples (data/benchmark/)
              ▼
         Embeddings ─▶ Storage (Local / pgvector / FAISS) ─▶ retrieve
                                                                │
question ───────────────────────────────────────────▶ Prompt.build_prompt
                                                                │
                                                     Execution.generate_funsql
                                                                │
                                         Execution.sanity_check_funsql (parse → build → render → run)
                                                                │
                                                     Evaluation (scores against gold SQL)
```

`Query.generate_funsql_query` is a second, shorter path that hands retrieval and
generation to `RAGTools.airag`. It shares `FUNSQL_SYSTEM_PROMPT` with the
explicit path, but the explicit path is the one the evaluation uses: every stage
is a separate function, so a failure can be attributed to retrieval, generation,
or the query itself.

## Design decisions

**Chunk by content type, not with one splitter.** A FunSQL example is a
`(question, SQL, FunSQL)` record, an OMOP table is a heading plus its column
list, and prose is prose. `RecordChunk`, `HeaderChunk` and `RecursiveChunk` keep
each of those whole, and fenced code is never split mid-expression. Splitting a
table's column list across two chunks is how a model ends up inventing columns.

**Ground strictly, and treat the context as the only source of names.**
`FUNSQL_SYSTEM_PROMPT` tells the model to use only tables, columns and constructs
that appear in the retrieved context, and to say what is missing rather than
guess. The zero-shot baseline prompt asks for the same output but drops those
rules. Otherwise the baseline would be told to refuse when it has no context,
and the comparison would measure the instruction rather than the retrieval.

**Verify the grounding material itself.** The docs in `data/grounding/` are what
the model is told to copy from. The test suite renders every FunSQL example in
them against a catalog built from the schema doc, and executes it on an empty
DuckDB database with that schema. This has caught examples using constructs
FunSQL doesn't have (`Distinct`, `Offset`), a join alias used after `Group`, and
`.&`, which FunSQL renders as bitwise `&` and DuckDB rejects on booleans.

**Score by execution, not by text.** Two correct FunSQL queries can look nothing
alike. The evaluation renders a generated query against the real schema, runs
it, and compares its result with the gold SQL's result. `results_match` aligns
columns by content rather than alias, ignores row order, and compares numbers
with a tolerance. Tasks whose gold result is empty are excluded, because any query
that returns nothing would match them.

**Keep heavy dependencies optional.** FunSQL and FAISS are only needed by some
users, so HealthLLM looks them up in `Main` (`require_main_module`) instead of
depending on them. The evaluation harness reaches the database, the model and the
retriever only through functions passed in, so it runs offline in the test suite
with a fake model and database.

**Target one SQL dialect at a time.** The benchmark and grounding target DuckDB.
FunSQL itself is dialect-independent, but the SQL functions a query calls are not.
`Fun.getdate()` is fine for SQL Server and an error on DuckDB. Examples written for
one dialect teach those function names to a pipeline targeting another, so the
retrievable examples must match the target.

**Hold evaluated tasks out of the index.** `split_tasks` makes a deterministic,
tier-stratified split. Only the other half's FunSQL is indexed, so no task can
retrieve its own answer.

## Known limitations

- **Generated code is executed.** `sanity_check_funsql` `eval`s the model's code
  to build the query. The default sandbox module limits name lookup, not what the
  code can do. Only run it on output from models you trust, on a machine you
  don't mind them running code on. The benchmark attaches the database read-only.
- **One shot, no repair.** The model gets one attempt per question. The
  `FunSQLCheck` error from a failed render or execution is not fed back for a
  second try, though that is often enough to fix a wrong column name.
- **Synthetic data.** The benchmark runs on Synthea, which leaves several OMOP
  fields empty (costs, quantities, refills, stop reasons, care sites) and has one
  observation period per person. Tasks that need those were dropped, so the
  benchmark under-represents cost and utilization questions.
- **Benchmark size.** 180 tasks, half evaluated per run, from a single
  temperature-0 generation per task. Small differences between configurations
  are within noise. `compare_modes` reports an exact McNemar p-value so this is
  visible.
- **HTML sources lose table structure.** `html_to_text` flattens HTML tables to
  prose, so a rendered page such as the OMOP `cdm54.html` in `CURATED_SOURCES`
  does not chunk one table per chunk. The local Markdown schema doc does.
- **Dependency ceiling.** RAGTools 0.10 pins PromptingTools to 0.87, so newer
  PromptingTools releases can't be used until RAGTools supports them.
- **Ollama-only runner.** The benchmark script registers Ollama models. Other
  providers work through PromptingTools schemas (`register_models`) but haven't
  been benchmarked.

## Extension points

- **Self-correction:** on a failed `FunSQLCheck`, append the error to the
  conversation and generate again, with a small attempt budget. The harness can
  measure the gain directly.
- **Retrieval:** multi-query retrieval, re-ranking or hybrid keyword/vector
  search (#33). A retriever is just `question -> hits`, so these drop into
  `run_benchmark` unchanged.
- **Other dialects:** benchmark another database by writing an executor and
  catalog for it (`scripts/benchmark/omop_duckdb.jl` is the DuckDB version),
  together with grounding docs and examples in that dialect.
- **Guardrails and agentic workflows** (#2, #3): the `Execution` stages are the
  natural place to put checks on a query before it runs.
