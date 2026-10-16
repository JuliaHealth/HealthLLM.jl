# FunSQL benchmark tasks

`funsql_tasks.jsonl` holds 65 natural-language → FunSQL tasks over the OMOP CDM.
`HealthLLM.Evaluation` and `scripts/benchmark/run_benchmark.jl` use it. Each line
has these fields:

| Field | Meaning |
|-------|---------|
| `id` | Stable id: `<group>-<FNV-1a hash of the question>` |
| `query` | The natural-language question |
| `group` | OMOP area (`drug_exposure`, `condition_era`, ...) |
| `tier` | `simple`, `compositional`, or `schema_aware` (see `classify_tier`) |
| `description` | Longer description of the task |
| `sql_query` | Gold SQL (DuckDB dialect, tables as `cdm.<table>`). Its result defines the correct answer |
| `response` | Gold FunSQL, only when it was verified to return the gold SQL's result. Otherwise empty |

## Provenance

Curated from the 200-row FunSQL example dataset (`FunSQLQueries/train.jsonl`,
MIT) with:

```bash
julia --project=scripts/benchmark scripts/benchmark/curate_tasks.jl \
    FunSQLQueries/train.jsonl synthea_1M_3YR.duckdb data/benchmark/funsql_tasks.jsonl
```

The script ran against `synthea_1M_3YR.duckdb` (1,145,357 persons) with a
60-second timeout per query. Run it on its own: concurrent DuckDB workloads on
the same machine can push gold queries into out-of-memory failures and change
which tasks are kept.

| | Tasks |
|-|------:|
| Source records | 200 |
| Gold SQL fails on DuckDB | 114 |
| Gold SQL returns no rows | 20 |
| Gold SQL times out | 1 |
| **Kept** | **65** (simple 17, compositional 8, schema-aware 40) |
| Gold FunSQL verified | 1 |

Porting the failing SQL Server queries to DuckDB and repairing the dataset's
FunSQL would grow both the task set and the pool of retrievable examples.
