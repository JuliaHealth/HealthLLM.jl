# NormalizedTextValidator

"""
    NormalizedTextValidator(;
        lowercase::Bool=true,
        trim::Bool=true,
        normalize_newlines::Bool=true,
        collapse_whitespace::Bool=true,
        strip_punctuation::Bool=false
    )

Validates text output against expected text after applying deterministic normalization steps.

# Normalization options
- `lowercase`: Converts both texts to lowercase (default `true`).
- `trim`: Strips leading and trailing whitespace (default `true`).
- `normalize_newlines`: Converts `\\r\\n` to `\\n` (default `true`).
- `collapse_whitespace`: Collapses multiple whitespace characters to a single space (default `true`).
- `strip_punctuation`: Removes ASCII punctuation marks (default `false`).

# Examples
```julia
v = NormalizedTextValidator()
validate(v, " Hypertension is high blood pressure. ", "hypertension is high blood pressure.") # passed
```
"""
struct NormalizedTextValidator <: AbstractValidator
    lowercase::Bool
    trim::Bool
    normalize_newlines::Bool
    collapse_whitespace::Bool
    strip_punctuation::Bool

    function NormalizedTextValidator(;
        lowercase::Bool=true,
        trim::Bool=true,
        normalize_newlines::Bool=true,
        collapse_whitespace::Bool=true,
        strip_punctuation::Bool=false
    )
        new(lowercase, trim, normalize_newlines, collapse_whitespace, strip_punctuation)
    end
end

"""
    normalize_text(v::NormalizedTextValidator, text::AbstractString)::String

Applies the configured normalization rules of `v` to `text`.
"""
function normalize_text(v::NormalizedTextValidator, text::AbstractString)::String
    s = String(text)
    if v.normalize_newlines
        s = replace(s, "\r\n" => "\n", "\r" => "\n")
    end
    if v.lowercase
        s = Base.Unicode.lowercase(s)
    end
    if v.strip_punctuation
        s = replace(s, r"[^\w\s]" => "")
    end
    if v.collapse_whitespace
        s = replace(s, r"\s+" => " ")
    end
    if v.trim
        s = strip(s)
    end
    return s
end

function validate(v::NormalizedTextValidator, actual, expected)::ValidationResult
    if !(actual isa AbstractString) || !(expected isa AbstractString)
        return ValidationResult(
            false,
            "NormalizedTextValidator requires string inputs, got actual=$(typeof(actual)), expected=$(typeof(expected))";
            expected=expected,
            actual=actual
        )
    end

    norm_actual = normalize_text(v, actual)
    norm_expected = normalize_text(v, expected)

    passed = norm_actual == norm_expected
    if passed
        return ValidationResult(
            true,
            "Normalized text match";
            expected=expected,
            actual=actual,
            details=Dict(:norm_actual => norm_actual, :norm_expected => norm_expected)
        )
    else
        return ValidationResult(
            false,
            "Normalized text mismatch.\nExpected (normalized): \"$norm_expected\"\nActual (normalized): \"$norm_actual\"";
            expected=expected,
            actual=actual,
            details=Dict(:norm_actual => norm_actual, :norm_expected => norm_expected)
        )
    end
end
