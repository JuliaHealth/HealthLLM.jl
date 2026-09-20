module TestHarness

using Test
using JSON3
using DataFrames
using DuckDB
using FunSQL
using PromptingTools
using RAGTools
using LinearAlgebra
using SparseArrays

include("types.jl")
include("validators/base.jl")
include("validators/exact.jl")
include("validators/normalized_text.jl")
include("validators/content.jl")
include("validators/structured.jl")
include("validators/execution.jl")
include("executors.jl")
include("mocking.jl")
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
export MockLLM, MockHealthLLMSchema, create_mock_rag_index
export load_golden_cases, save_golden_cases, create_test_corpus, create_test_duckdb
export @test_valid

"""
    @test_valid validator actual [expected]

A testing macro that evaluates `validate(validator, actual, expected)` within standard Julia `@test`.
Provides detailed failure messages if the validation fails.
"""
macro test_valid(validator, actual, expected=nothing)
    return quote
        val_res = validate($(esc(validator)), $(esc(actual)), $(esc(expected)))
        if !val_res.passed
            @error "Validation failure: " * val_res.message
        end
        @test val_res.passed
    end
end

"""
    run_test_suite(cases::Vector{TestCase}; verbose::Bool=true)::Vector{TestResult}

Runs a sequence of `TestCase`s and displays summarized execution statistics.
"""
function run_test_suite(cases::Vector{TestCase}; verbose::Bool=true)::Vector{TestResult}
    results = TestResult[]
    passed_count = 0

    for tc in cases
        res = run_test_case(tc)
        push!(results, res)
        if res.validation_result.passed
            passed_count += 1
            if verbose
                println("  ✓ [$(tc.category)] $(tc.name) ($(round(res.execution_time * 1000, digits=2)) ms)")
            end
        else
            if verbose
                println("  ✗ [$(tc.category)] $(tc.name): $(res.validation_result.message)")
            end
        end
    end

    if verbose
        println("\nSuite Summary: $passed_count / $(length(cases)) passed.")
    end

    return results
end

end # module TestHarness
