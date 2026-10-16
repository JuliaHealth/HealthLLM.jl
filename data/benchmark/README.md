# FunSQL benchmark tasks

Natural-language → FunSQL tasks over the OMOP CDM, each verified by execution on
the Synthea 1M DuckDB database. `HealthLLM.Evaluation` and
`scripts/benchmark/run_benchmark.jl` use them.

| File | Contents |
|------|----------|
| `funsql_duckdb_source.jsonl` | The repaired task set (180 rows), with provenance. Edit this one. |
| `funsql_tasks.jsonl` | The benchmark file generated from it by `curate_tasks.jl`. Don't edit by hand. |

## Fields

`funsql_tasks.jsonl`:

| Field | Meaning |
|-------|---------|
| `id` | Stable id: `<group>-<FNV-1a hash of the question>` |
| `query` | The natural-language question. It names every concept id, code and filter needed to answer it |
| `group` | OMOP area (`drug_exposure`, `condition_era`, ...) |
| `tier` | `simple`, `compositional`, or `schema_aware` (see `classify_tier`) |
| `description` | Same as `query` |
| `sql_query` | Gold SQL (DuckDB dialect, tables as `cdm.<table>`). Its result defines the correct answer |
| `response` | Gold FunSQL, present only when verified to return the gold SQL's result |

`funsql_duckdb_source.jsonl` has the same task fields plus `source_index` (the
line in the original `FunSQLQueries/train.jsonl`), `source_query` (the original
question), and `note` (how the row was derived: `ported`, `adapted: ...`,
`rewritten to match the question`, ...).

## How the tasks were built

The original 200-row FunSQL example dataset (`FunSQLQueries/train.jsonl`, MIT,
derived from the OHDSI QueryLibrary) couldn't be used as-is on DuckDB with Synthea:

- most gold SQL was SQL Server dialect (`GETDATE`, `DATEADD`, `DATEDIFF(DAY, …)`,
  `STDEV`);
- many questions didn't match their SQL. From record 27 to 41, for example, the
  questions are shifted one record against the SQL;
- questions left out the concept ids the SQL depended on, and some SQL still had
  unfilled `@person_id`-style parameters;
- most input concepts (e.g. abdominal pain 200219, sciatica 372409, rosuvastatin)
  never occur in Synthea, so faithful ports returned empty results.

Each record was therefore rebuilt as a self-contained task: a question that says
exactly what to compute, DuckDB SQL, and a FunSQL translation. Concepts absent
from Synthea were swapped for ones that occur, keeping the analytical pattern
(`note` records each change). Every pair was checked with
`scripts/benchmark/verify_pairs.jl`, which runs both queries and compares the results.
Then `curate_tasks.jl` re-ran everything to produce `funsql_tasks.jsonl`:

```bash
julia --project=scripts/benchmark scripts/benchmark/curate_tasks.jl \
    data/benchmark/funsql_duckdb_source.jsonl synthea_1M_3YR.duckdb data/benchmark/funsql_tasks.jsonl
```

Run curation on its own. Concurrent DuckDB workloads on the same machine can push
large queries into out-of-memory failures and change which tasks are kept.

### Dropped records (20)

| Records | Reason |
|---------|--------|
| 62, 63, 64, 66 | All cost amounts are NULL in Synthea |
| 36, 102, 123 | `stop_reason` is empty in Synthea |
| 119, 120 | `drug_exposure.quantity` is empty in Synthea |
| 121, 122 | `drug_exposure.refills` is empty in Synthea |
| 106, 141, 142, 144, 145 | Indication and contraindication relationships are not in this vocabulary |
| 171, 175, 176 | Every Synthea person has exactly one observation period |
| 177 | Its month-spreading calculation has no faithful single-query equivalent |

## Relation to the Hugging Face dataset

[`JuliaHealthOrg/FunSQLQueries`](https://huggingface.co/datasets/JuliaHealthOrg/FunSQLQueries)
was repaired independently upstream. It keeps all 200 rows with **SQL Server**
as the target dialect and verifies them on a seeded synthetic database. Its FunSQL
calls SQL Server functions (`Fun.getdate`, `Fun.dateadd`, `Fun.stdev`), which
DuckDB rejects. So it is the reference NL → FunSQL dataset, while this set is the
DuckDB/Synthea-executable benchmark HealthLLM evaluates against. Don't index
FunSQL written for one dialect as RAG examples for a pipeline targeting another:
the model copies the function names.
