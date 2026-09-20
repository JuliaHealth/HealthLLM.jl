# ExactValidator

"""
    ExactValidator(; atol::Union{Nothing,Real}=nothing, rtol::Union{Nothing,Real}=nothing)

Validates that the actual output exactly matches the expected output using `==` (or `isapprox` if tolerances are provided).

# Examples
```julia
v = ExactValidator()
validate(v, 42, 42) # passed

v_approx = ExactValidator(atol=0.01)
validate(v_approx, 1.005, 1.0) # passed
```
"""
struct ExactValidator <: AbstractValidator
    atol::Union{Nothing,Float64}
    rtol::Union{Nothing,Float64}

    function ExactValidator(; atol=nothing, rtol=nothing)
        new(
            atol === nothing ? nothing : Float64(atol),
            rtol === nothing ? nothing : Float64(rtol)
        )
    end
end

function validate(v::ExactValidator, actual, expected)::ValidationResult
    # Numeric approximate comparison
    if (v.atol !== nothing || v.rtol !== nothing) && actual isa Number && expected isa Number
        atol = v.atol === nothing ? 0.0 : v.atol
        rtol = v.rtol === nothing ? (atol > 0 ? 0.0 : √eps(Float64)) : v.rtol
        passed = isapprox(actual, expected; atol=atol, rtol=rtol)
        msg = passed ? "Exact (approx) match: $actual ≈ $expected" :
                       "Value mismatch: expected ≈ $expected, got $actual (atol=$(v.atol), rtol=$(v.rtol))"
        return ValidationResult(passed, msg; expected=expected, actual=actual)
    end

    # Array approximate comparison
    if (v.atol !== nothing || v.rtol !== nothing) && actual isa AbstractArray{<:Number} && expected isa AbstractArray{<:Number}
        atol = v.atol === nothing ? 0.0 : v.atol
        rtol = v.rtol === nothing ? (atol > 0 ? 0.0 : √eps(Float64)) : v.rtol
        passed = size(actual) == size(expected) && isapprox(actual, expected; atol=atol, rtol=rtol)
        msg = passed ? "Array (approx) match" :
                       "Array mismatch: expected ≈ $expected, got $actual"
        return ValidationResult(passed, msg; expected=expected, actual=actual)
    end

    # Standard exact comparison
    passed = isequal(actual, expected)
    if passed
        return ValidationResult(true, "Exact match"; expected=expected, actual=actual)
    else
        return ValidationResult(
            false,
            "Exact match failed. Expected: $(repr(expected)), Actual: $(repr(actual))";
            expected=expected,
            actual=actual
        )
    end
end
