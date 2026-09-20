# Core types for HealthLLM Modular Testing Harness

"""
    AbstractValidator

Abstract supertype for all test validators. Subtypes must implement:
`validate(validator::AbstractValidator, actual, expected)::ValidationResult`
"""
abstract type AbstractValidator end

"""
    AbstractExecutor

Abstract supertype for all test executors. Subtypes must implement:
`execute(executor::AbstractExecutor, input)`
"""
abstract type AbstractExecutor end

"""
    ValidationResult

Encapsulates the outcome of validating actual output against expected criteria.

# Fields
- `passed::Bool`: Whether the validation passed.
- `message::String`: Diagnostic message explaining pass or failure reason.
- `expected::Any`: Expected value or criteria.
- `actual::Any`: Observed value.
- `details::Dict{Symbol,Any}`: Additional diagnostic metadata (e.g. diffs, field errors).
"""
struct ValidationResult
    passed::Bool
    message::String
    expected::Any
    actual::Any
    details::Dict{Symbol,Any}

    function ValidationResult(passed::Bool, message::String; expected=nothing, actual=nothing, details=Dict{Symbol,Any}())
        new(passed, message, expected, actual, details)
    end
end

"""
    TestCase

Represents a single test case with an input, expected output/behavior, a validator,
and an optional executor and metadata.

# Fields
- `name::String`: Unique identifier or name of the test case.
- `input::Any`: Input data/prompt/arguments.
- `expected::Any`: Expected output or validation criteria.
- `validator::AbstractValidator`: The validator strategy to evaluate the output.
- `executor::Union{Nothing, AbstractExecutor}`: Optional executor to run the test.
- `category::Symbol`: Category tag (e.g. `:unit`, `:llm`, `:golden`, `:execution`, `:integration`).
- `metadata::Dict{Symbol,Any}`: Arbitrary metadata (e.g. description, tags, timeout).
"""
struct TestCase
    name::String
    input::Any
    expected::Any
    validator::AbstractValidator
    executor::Union{Nothing, AbstractExecutor}
    category::Symbol
    metadata::Dict{Symbol,Any}

    function TestCase(
        name::String,
        input::Any,
        expected::Any,
        validator::AbstractValidator;
        executor::Union{Nothing, AbstractExecutor}=nothing,
        category::Symbol=:unit,
        metadata::Dict{Symbol,Any}=Dict{Symbol,Any}()
    )
        new(name, input, expected, validator, executor, category, metadata)
    end
end

"""
    TestResult

Result of executing and validating a `TestCase`.

# Fields
- `test_case::TestCase`: The executed test case.
- `validation_result::ValidationResult`: The outcome of validation.
- `execution_time::Float64`: Elapsed time in seconds.
- `error::Union{Nothing, Exception}`: Any unhandled exception during execution.
"""
struct TestResult
    test_case::TestCase
    validation_result::ValidationResult
    execution_time::Float64
    error::Union{Nothing, Exception}

    function TestResult(test_case::TestCase, validation_result::ValidationResult; execution_time::Float64=0.0, error=nothing)
        new(test_case, validation_result, execution_time, error)
    end
end

# Utility methods
Base.show(io::IO, r::ValidationResult) = print(io, "ValidationResult(passed=$(r.passed), message=\"$(r.message)\")")
Base.show(io::IO, t::TestCase) = print(io, "TestCase(\"$(t.name)\", category=:$(t.category), validator=$(typeof(t.validator)))")
