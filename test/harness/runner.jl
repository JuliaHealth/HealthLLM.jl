# Test Harness Suite Runner

export run_test_suite

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
