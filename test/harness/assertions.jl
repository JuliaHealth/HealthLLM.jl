# Test Harness Assertion Helpers

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
