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
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      Function         Mock LLM         FunSQL / SQL
      Execution        Execution         Execution
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                        Output
                           │
                           ▼
                       Validator
   ┌───────────────┬───────┴───────┬───────────────┐
   ▼               ▼               ▼               ▼
 Exact        Normalized       Required          JSON
 Match           Text           Content        Structure
   │               │               │               │
   └───────────────┼───────────────┴───────────────┘
                   ▼
            ValidationResult
          (passed, message, details)
```

---

## 2. Running Tests

### Standard Test Run
Run the full offline test suite via Julia's package manager:

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
# Run only deterministic unit tests & harness self-tests
HEALTHLLM_TEST_CATEGORY=unit julia --project=. -e 'using Pkg; Pkg.test()'

# Run only LLM & mock tests
HEALTHLLM_TEST_CATEGORY=llm julia --project=. -e 'using Pkg; Pkg.test()'

# Run only golden reference regression tests
HEALTHLLM_TEST_CATEGORY=golden julia --project=. -e 'using Pkg; Pkg.test()'

# Run only SQL / FunSQL execution tests
HEALTHLLM_TEST_CATEGORY=execution julia --project=. -e 'using Pkg; Pkg.test()'

# Run only end-to-end integration tests
HEALTHLLM_TEST_CATEGORY=integration julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## 3. Available Validators

The framework includes standard validators that can be used directly or inside the `@test_valid` macro:

### `ExactValidator`
For deterministic functions where exact equality (or numeric approximation) is expected:

```julia
using Test
using TestHarness

@test_valid ExactValidator() (10 + 20) 30
@test_valid ExactValidator(atol=0.01) 1.005 1.000
```

### `NormalizedTextValidator`
For textual outputs that may differ in minor formatting (casing, surrounding whitespace, newlines, or punctuation):

```julia
v = NormalizedTextValidator(lowercase=true, trim=true, collapse_whitespace=true)
@test_valid v "  Hypertension is high blood pressure. \n" "hypertension is high blood pressure."
```

### `ContainsValidator` & `RequiredContentValidator`
For generated clinical text where specific concepts or phrases must appear (and forbidden claims must not):

```julia
v = RequiredContentValidator(
    required=["elevated blood pressure", "systolic", "diastolic"],
    forbidden=["cure guaranteed"],
    mode=:all # :all requires all items, :any requires at least one
)

response = "Hypertension is defined as elevated blood pressure with systolic >= 130."
@test_valid v response
```

### `JSONStructureValidator`
For structured LLM extractions, validating JSON schema, types, allowed value ranges, and enums:

```julia
jv = JSONStructureValidator(
    required_keys=["patient_id", "condition", "confidence"],
    key_types=Dict("patient_id" => Integer, "condition" => AbstractString),
    range_constraints=Dict("confidence" => (0.0, 1.0)),
    allowed_values=Dict("urgency" => ["low", "moderate", "high"])
)

raw_json = """{"patient_id": 1, "condition": "hypertension", "confidence": 0.95, "urgency": "moderate"}"""
@test_valid jv raw_json
```

### `ExecutionValidator`
For generated SQL or FunSQL queries, executing them against a test database (e.g. in-memory DuckDB) and asserting on the returned `DataFrame`:

```julia
db = create_test_duckdb()
ev = ExecutionValidator(db)

# Validate that generated SQL returns the expected DataFrame
generated_sql = "SELECT patient_id, first_name FROM patients WHERE age > 65 ORDER BY patient_id;"
expected_df = DataFrame(patient_id=[1, 3], first_name=["Alice", "Charlie"])

@test_valid ev generated_sql expected_df
```

---

## 4. Deterministic Mocking (`MockLLM` & `MockHealthLLMSchema`)

Tests in the standard test suite must never depend on external network calls or paid API tokens. Use `MockLLM` and `MockHealthLLMSchema` to simulate model responses:

```julia
using PromptingTools
using TestHarness

# Define mapping from prompt patterns to canned responses
mock = MockLLM(Dict(
    "hypertension" => "Hypertension is persistently elevated blood pressure.",
    r"diabetes.*treatment" => "First line treatment is metformin."
); default_response="Standard clinical response.")

schema = MockHealthLLMSchema(mock)

# Call PromptingTools offline
msg = PromptingTools.aigenerate(schema, "Explain hypertension")
@test_valid ContainsValidator("blood pressure") msg.content
```

---

## 5. Working with Golden References

Golden references represent version-controlled pairs of known inputs and expected behaviors stored under `test/fixtures/`.

### Directory Layout
```
test/fixtures/
├── golden_query_cases.json   # SQL / FunSQL golden query cases
├── golden_text_cases.json    # Clinical Q&A golden responses & criteria
├── structured_cases.json     # Structured JSON extraction golden cases
└── sample_corpus/            # Synthetic clinical notes (.txt, .md)
```

### Loading Golden Cases in Tests
```julia
using TestHarness

cases = load_golden_cases("test/fixtures/golden_text_cases.json")
for case in cases
    validator = RequiredContentValidator(required=case["required_concepts"])
    @test_valid validator case["mock_response"]
end
```

### Updating Golden References
If model behavior or requirements intentionally change:
1. Inspect the test failure diff to ensure the new behavior is desired.
2. Edit the corresponding fixture in `test/fixtures/*.json` (or use `save_golden_cases`).
3. Commit the updated reference along with the code change.

---

## 6. Security and Synthetic Data Guidelines

When writing fixtures or tests:
1. **Never use real Protected Health Information (PHI) or patient records.** All fixtures must use synthetic or anonymized public data.
2. **Never commit API keys or credentials.**
3. **Execution safety**: Always execute generated code against temporary, in-memory databases (`create_test_duckdb()`), never against production or external systems.
