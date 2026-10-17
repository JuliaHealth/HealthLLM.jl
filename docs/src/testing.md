# Modular Testing Harness & Contributor Guide

`HealthLLM.jl` provides a modular testing harness (`TestHarness`) located under `test/harness/` that cleanly separates **test case definition**, **execution**, and **validation**.

Because `HealthLLM.jl` combines deterministic software components (file parsers, database converters) with nondeterministic components (LLM prompts, RAG retrieval, query generation), different validation strategies are required for different outputs.
---

## 1. Core Architecture

```
TestCase(name, input, expected, validator, [executor], [metadata])
                           │
                           ▼
                    Test Executor
           ┌───────────────┴───────────────┐
           ▼                               ▼
       Function                       FunSQL / SQL
       Execution                       Execution
           │                               │
           └───────────────┬───────────────┘
                           ▼
                        Output
                           │
                           ▼
                       Validator
   ┌───────────────┬───────┴───────┬───────────────┬───────────────┐
   ▼               ▼               ▼               ▼               ▼
 Exact        Normalized       Required          JSON          Execution
 Match           Text           Content        Structure       Validation
   │               │               │               │               │
   └───────────────┴───────┬───────┴───────────────┴───────────────┘
                           ▼
                    ValidationResult
                  (passed, message, details)
```

---

## 2. Running Tests

### Standard Test Run
Run the full test suite via Julia's package manager:

```julia
using Pkg
Pkg.test("HealthLLM")
```

Or from the command line:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Running Specific Test Categories
You can run a specific test suite using the `HEALTHLLM_TEST_CATEGORY` environment variable:

```bash
# Run core upstream test suites
HEALTHLLM_TEST_CATEGORY=core julia --project=. -e 'using Pkg; Pkg.test()'

# Run deterministic harness self-tests
HEALTHLLM_TEST_CATEGORY=unit julia --project=. -e 'using Pkg; Pkg.test()'

# Run SQL / FunSQL execution tests
HEALTHLLM_TEST_CATEGORY=execution julia --project=. -e 'using Pkg; Pkg.test()'

# Run golden reference regression tests
HEALTHLLM_TEST_CATEGORY=golden julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## 3. Available Validators

The framework includes standard validators that can be evaluated directly via `validate(v, actual, expected)` or inside standard `@test`:

### `ExactValidator`
For deterministic functions where exact equality (or numeric approximation) is expected:

```julia
using Test
using TestHarness

v = ExactValidator()
res = validate(v, 10 + 20, 30)
@test res.passed

v_approx = ExactValidator(atol=0.01)
res_approx = validate(v_approx, 1.005, 1.000)
@test res_approx.passed
```

### `NormalizedTextValidator`
For textual outputs that may differ in minor formatting (casing, surrounding whitespace, newlines, or punctuation):

```julia
using Test
using TestHarness

v = NormalizedTextValidator(lowercase=true, trim=true, collapse_whitespace=true)
res = validate(v, "  Hypertension is high blood pressure. \n", "hypertension is high blood pressure.")
@test res.passed
```

### `ContainsValidator` & `RequiredContentValidator`
For generated clinical text where specific concepts or phrases must appear (and forbidden claims must not):

```julia
using Test
using TestHarness

v = RequiredContentValidator(
    required=["elevated blood pressure", "systolic", "diastolic"],
    forbidden=["cure guaranteed"],
    mode=:all # :all requires all items, :any requires at least one
)

response = "Hypertension is defined as elevated blood pressure with systolic >= 130 or diastolic >= 80."
res = validate(v, response)
@test res.passed
```

### `JSONStructureValidator`
For structured extractions, validating JSON schema, types, allowed value ranges, and enums:

```julia
using Test
using TestHarness

jv = JSONStructureValidator(
    required_keys=["patient_id", "condition", "confidence"],
    key_types=Dict("patient_id" => Integer, "condition" => AbstractString),
    range_constraints=Dict("confidence" => (0.0, 1.0)),
    allowed_values=Dict("urgency" => ["low", "moderate", "high"])
)

raw_json = """{"patient_id": 1, "condition": "hypertension", "confidence": 0.95, "urgency": "moderate"}"""
res = validate(jv, raw_json)
@test res.passed
```

### `ExecutionValidator`
For generated SQL or FunSQL queries, executing them against a database connection (e.g. in-memory DuckDB) or executor function and comparing against an expected `DataFrame` or `ResultSet`:

```julia
using Test
using DataFrames
using TestHarness

db = create_test_duckdb()
ev = ExecutionValidator(db)

# Validate that generated SQL returns the expected DataFrame
generated_sql = "SELECT patient_id, first_name FROM patients WHERE age > 65 ORDER BY patient_id;"
expected_df = DataFrame(patient_id=[1, 3], first_name=["Alice", "Charlie"])

res = validate(ev, generated_sql, expected_df)
@test res.passed
```

`ExecutionValidator` also supports schema catalogs and dialect preservation:

```julia
using Test
using DataFrames
using FunSQL
using TestHarness

db = create_test_duckdb()
catalog = FunSQL.SQLCatalog(
    FunSQL.SQLTable(:patients, columns=[:patient_id, :first_name, :age]);
    dialect=:duckdb
)
ev_cat = ExecutionValidator(db; catalog=catalog)

# Selects patient_id and first_name for patients aged 65 and older
funsql_code = "From(:patients) |> Where(Fun.\">=\"(Get.age, 65)) |> Select(Get.patient_id, Get.first_name)"
expected_df = DataFrame(patient_id=[1, 3], first_name=["Alice", "Charlie"])

res = validate(ev_cat, funsql_code, expected_df)
@test res.passed
```

---

## 4. Testing with Injected Generators and Executors

`HealthLLM.jl` is designed with native injection interfaces. Harness tests should use injected generators and executors to avoid unnecessary external network calls and paid API dependencies. Instead of mocking network libraries, use callable generator and executor injections:

### Injectable `generate_funsql`
Pass a generator function `(prompt; model, schema, kwargs...) -> reply`:

```julia
using Test
using HealthLLM

fake_gen = (prompt; kwargs...) -> "```julia\nFrom(:person) |> Select(:person_id)\n```\nCounts persons."
gen = HealthLLM.Execution.generate_funsql("prompt"; generator=fake_gen)
@test gen.funsql == "From(:person) |> Select(:person_id)"
```

### Injectable `run_task`
Pass an injectable `generator`, `executor`, and `catalog`:

```julia
using Test
using FunSQL
using HealthLLM.Evaluation: BenchmarkTask, run_task, ResultSet

task = BenchmarkTask("task-1", "How many patients?", "patients", :simple, "SELECT COUNT(*) AS n FROM patients", "", "")
test_catalog = FunSQL.SQLCatalog(FunSQL.SQLTable(:patients, columns=[:patient_id]))
mock_executor = sql -> ResultSet(["n"], [[42]])
mock_gen = (prompt; kwargs...) -> "```julia\nFrom(:patients) |> Group() |> Select(:n => Agg.count())\n```"

result = run_task(task, :zero_shot; model="fake", executor=mock_executor, catalog=test_catalog, generator=mock_gen)
@test result.outcome === :correct
```

---

## 5. Working with Golden References & Validation Fixtures

Validation fixtures represent version-controlled pairs of known inputs and expected behaviors stored under `test/fixtures/`.

### Directory Layout
```
test/fixtures/
├── golden_query_cases.json   # SQL test queries & expected patient/observation IDs
├── golden_text_cases.json    # Clinical text responses & required concepts
├── structured_cases.json     # Structured JSON extraction fixtures
└── sample_corpus/            # Synthetic clinical notes (.txt, .md)
```

### Loading Fixtures in Tests
```julia
using Test
using TestHarness

cases = load_golden_cases("test/fixtures/golden_text_cases.json")
for case in cases
    validator = RequiredContentValidator(required=case["required_concepts"])
    res = validate(validator, case["mock_response"])
    @test res.passed
end
```

### Updating Golden References

When expected behavior intentionally changes, update the relevant fixtures carefully:

1. **Investigate the failure.** Determine whether it reveals a regression or an intentional behavior change.
2. **Verify the new expected behavior.** For query tests, check that the generated query matches the intended database semantics. For text and structured-output tests, verify the expected content against the test requirements.
3. **Update the appropriate file.** Edit the corresponding JSON fixture under `test/fixtures/`. If an expectation is defined directly in `test/golden/test_golden_regression.jl`, update that test instead.
4. **Run the regression tests.** Confirm that the updated case passes and that negative regression checks still detect incorrect behavior.
5. **Run the full test suite** when practical to check for unintended effects.

Run the golden regression tests directly:

```bash
julia --project=test test/golden/test_golden_regression.jl
```

Or run them through Julia's package test runner:

```bash
HEALTHLLM_TEST_CATEGORY=golden julia --project=. -e 'using Pkg; Pkg.test()'
```

Commit fixture changes alongside the code change that justifies them. **Never change expected results solely to make a failing test pass.**

---

## 6. Security and Synthetic Data Guidelines

When writing fixtures or tests:
1. **Never use real Protected Health Information (PHI) or patient records.** All fixtures must use synthetic or anonymized public data.
2. **Never commit API keys or credentials.**
3. **Execution safety**: Always execute generated code against temporary, in-memory databases (`create_test_duckdb()`), never against production or external systems.
