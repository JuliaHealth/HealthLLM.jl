# Base interface for Validators

"""
    validate(validator::AbstractValidator, actual, expected)::ValidationResult

Evaluate whether `actual` satisfies `expected` according to `validator`.
Must return a `ValidationResult`.
"""
function validate(validator::AbstractValidator, actual, expected)::ValidationResult
    error("Validator $(typeof(validator)) must implement validate(validator, actual, expected)")
end
