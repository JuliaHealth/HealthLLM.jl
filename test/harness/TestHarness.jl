module TestHarness

using Test
using JSON3
using DataFrames
using DuckDB
using FunSQL
using HealthLLM
using HealthLLM.Evaluation: ResultSet, results_match
using HealthLLM.Execution: sanity_check_funsql, extract_funsql, FunSQLGeneration, FunSQLCheck

include("types.jl")
include("validators/base.jl")
include("validators/exact.jl")
include("validators/normalized_text.jl")
include("validators/content.jl")
include("validators/structured.jl")
include("validators/execution.jl")
include("executors.jl")
include("assertions.jl")
include("runner.jl")
include("fixtures.jl")

# Exports
export AbstractValidator, AbstractExecutor
export ValidationResult, TestCase, TestResult
export validate, execute, run_test_case, run_test_suite
export ExactValidator
export NormalizedTextValidator, normalize_text
export ContainsValidator, RequiredContentValidator
export JSONStructureValidator
export ExecutionValidator
export FunctionExecutor, QueryExecutor
export load_golden_cases, save_golden_cases, create_test_corpus, create_test_duckdb
export ResultSet, results_match, sanity_check_funsql, extract_funsql
export @test_valid

end # module TestHarness
