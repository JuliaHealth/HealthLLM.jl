# End-to-End Example

`scripts/end_to_end.jl` takes one natural-language question through the whole
pipeline and prints what each stage did:

1. register the chat and embedding models and open the OMOP database read-only;
2. chunk and embed the grounding docs (`data/grounding/`) and the verified FunSQL
   examples (`data/benchmark/`), and put them in a [`LocalVectorStore`](@ref HealthLLM.Storage.LocalVectorStore);
3. retrieve the chunks closest to the question;
4. build the grounded prompt with [`build_prompt`](@ref HealthLLM.Prompt.build_prompt);
5. generate FunSQL with [`generate_funsql`](@ref HealthLLM.Execution.generate_funsql);
6. parse, build and render it against the database schema with
   [`sanity_check_funsql`](@ref HealthLLM.Execution.sanity_check_funsql);
7. run the SQL and print the result table.

The corpus is small and ships with the repository: three Markdown grounding docs
(the OMOP CDM schema, a FunSQL reference, and SQL → FunSQL patterns) plus the
benchmark's verified examples, 239 chunks in all. Benchmark examples that ask the
exact question being run are left out of the index, so the model has to compose
the answer rather than look it up.

## Setup

You need three things.

**The script environment.** From the repository root:

```bash
julia --project=scripts/benchmark -e 'using Pkg; Pkg.instantiate()'
```

On Julia 1.10, also run `Pkg.develop(path=".")` in that environment, because the
`[sources]` entry pointing at the repository needs Julia 1.11 or later.

**Ollama** serving a chat model and the embedding model:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

**An OMOP CDM DuckDB file.** The examples and the benchmark use the Synthea 1M
database (`synthea_1M_3YR.duckdb`, about 19 GB) from the
[`JuliaHealthOrg/JuliaHealthDatasets`](https://huggingface.co/datasets/JuliaHealthOrg/JuliaHealthDatasets)
Hugging Face dataset. Save it in the repository root, where the scripts look by
default (it is gitignored):

```bash
curl -L -o synthea_1M_3YR.duckdb \
    https://huggingface.co/datasets/JuliaHealthOrg/JuliaHealthDatasets/resolve/main/synthea_1M_3YR.duckdb
```

The file is attached read-only. Generated queries can't modify it.

## Running it

```bash
julia --project=scripts/benchmark scripts/end_to_end.jl
```

| Option | Default |
|--------|---------|
| `--db PATH` | `synthea_1M_3YR.duckdb` in the repository root |
| `--model NAME` | `qwen2.5:7b` |
| `--question TEXT` | a hypertension-by-gender question (below) |

Embedding the 239 chunks takes a few minutes with a local Ollama (about 5 minutes
on a laptop CPU). After that, retrieval and generation take seconds.

## Reading the output

This is a real run with `qwen2.5:7b`. Long lines are cut at `...` and paths are
shown relative to the repository:

```
========================================================================
1. Models and database
========================================================================
chat model: qwen2.5:7b, embedding model: nomic-embed-text
database: synthea_1M_3YR.duckdb (39 OMOP tables, read-only)

========================================================================
2. Build the retrieval index
========================================================================
indexed 239 chunks from 4 documents

========================================================================
3. Retrieve context for the question
========================================================================
question: How many distinct patients had a condition era of essential hypertension
(concept 320128) starting in 2020, broken down by gender concept name?

[1]  0.691  benchmark:examples › Among patients with essential hypertension (a condition era of concept 320128), ...
[2]  0.684  benchmark:examples › What is the average length in days (visit end date minus visit start date) of ...
[3]  0.682  data/grounding/omop_cdm_schema.md › Common Query Patterns in OMOP
[4]  0.681  benchmark:examples › Count the distinct persons with at least one condition era of each of ...
[5]  0.665  benchmark:examples › How many condition occurrences of essential hypertension (concept 320128) ...
[6]  0.665  benchmark:examples › Define the cohort as patients with a type 2 diabetes mellitus ...

========================================================================
4. Build the grounded prompt
========================================================================
system prompt: 1795 characters (FUNSQL_SYSTEM_PROMPT)
user turn: 7161 characters (retrieved context + question)

========================================================================
5. Generate FunSQL
========================================================================
From(:condition_era) |>
Where(Fun.and(Get.condition_concept_id .== 320128, Fun.date_between(Get.condition_start_date, "2020-01-01", "2020-12-31"))) |>
Join(:person => From(:person), Get.person_id .== Get.person.person_id) |>
Join(:concept => From(:concept), Get.person.gender_concept_id .== Get.concept.concept_id) |>
Group(Get.concept.concept_name) |>
Select(Get.concept.concept_name, :num_patients => Agg.count_distinct(Get.person.person_id))

========================================================================
6. Check it against the schema
========================================================================
FunSQLCheck(failed @ :built, error="FunSQL.ReferenceError: cannot find `condition_start_date` in: ...")

The query did not pass the check, so it is not executed:
FunSQL.ReferenceError: cannot find `condition_start_date` in: ...
```

Each retrieval hit shows its similarity score and provenance: the source document
and the heading or example it came from. The retrieved context here is on topic.
It has three hypertension examples and the schema doc's query patterns.

The model still got the query wrong, and step 6 is where that shows up. In
`condition_era` the start column is `condition_era_start_date`.
`condition_start_date` belongs to `condition_occurrence`. Rendering against the
real schema catches the invented column before anything reaches the database.
(`Fun.date_between` would have failed next, because DuckDB has no such function.)
A failed check is a normal outcome, not a crash. The script prints the error and
stops.

Another common failure is using a join alias after `Group`. For example, the model
writes `Group(Get.c.concept_name)` and then `Select(Get.c.concept_name, ...)`.
After `Group`, only the grouping keys and aggregates are in scope, so the key
needs a name of its own. With the alias fixed, the same pipeline runs the query:

```julia
From(:condition_era) |>
Where(Get.condition_concept_id .== 320128) |>
Join(:person => From(:person), Get.person_id .== Get.person.person_id) |>
Join(:gender => From(:concept), Get.person.gender_concept_id .== Get.gender.concept_id) |>
Group(:gender_name => Get.gender.concept_name) |>
Select(Get.gender_name, :num_persons => Agg.count_distinct(Get.person_id))
```

```
FunSQLCheck(ok @ :rendered)

 Row │ gender_name  num_persons
     │ String       Int64
─────┼──────────────────────────
   1 │ FEMALE            143221
   2 │ MALE              146282
```

The model gets one attempt per question. [Design and Limitations](design.md)
describes feeding the check's error back for a second try.

## From one question to a benchmark

The walkthrough shows a single question. [Evaluation](evaluation.md) runs the same
stages over all 180 benchmark tasks, with and without retrieval, and scores every
generated query by executing it against the gold SQL's result:

```bash
julia --project=scripts/benchmark scripts/benchmark/run_benchmark.jl \
    --db synthea_1M_3YR.duckdb --model qwen2.5:7b
```
