# Content & Concept Validators

"""
    ContainsValidator(; pattern::Union{Nothing,AbstractString,Regex}=nothing, case_sensitive::Bool=false)

Validates that the output text contains a specific substring or matches a regular expression.
If `pattern` is not provided at construction, `expected` passed to `validate` will be used as the pattern.
"""
struct ContainsValidator <: AbstractValidator
    pattern::Union{Nothing,String,Regex}
    case_sensitive::Bool

    function ContainsValidator(pattern=nothing; case_sensitive::Bool=false)
        pat = pattern isa AbstractString ? String(pattern) : pattern
        new(pat, case_sensitive)
    end
end

function validate(v::ContainsValidator, actual, expected=nothing)::ValidationResult
    if !(actual isa AbstractString)
        return ValidationResult(false, "ContainsValidator requires string actual output, got $(typeof(actual))"; actual=actual)
    end

    target = v.pattern !== nothing ? v.pattern : expected
    if target === nothing
        return ValidationResult(false, "No pattern specified for ContainsValidator"; actual=actual)
    end

    passed = false
    if target isa Regex
        passed = occursin(target, actual)
    else
        act_str = v.case_sensitive ? String(actual) : lowercase(String(actual))
        tgt_str = v.case_sensitive ? String(target) : lowercase(String(target))
        passed = occursin(tgt_str, act_str)
    end

    if passed
        return ValidationResult(true, "Output contains expected pattern: $(repr(target))"; expected=target, actual=actual)
    else
        return ValidationResult(
            false,
            "Expected pattern $(repr(target)) not found in actual output: $(repr(actual))";
            expected=target,
            actual=actual
        )
    end
end

"""
    RequiredContentValidator(;
        required::Vector{String}=String[],
        forbidden::Vector{String}=String[],
        mode::Symbol=:all,
        case_sensitive::Bool=false
    )

Validates that text output contains required concepts or phrases and optionally does NOT contain forbidden concepts.

# Arguments
- `required`: List of required concepts/phrases. Can also be passed as `expected` to `validate`.
- `forbidden`: List of concepts that must NOT appear in `actual`.
- `mode`: `:all` (default) requires all items in `required` to be present; `:any` requires at least one.
- `case_sensitive`: Whether matching is case sensitive (default `false`).

# Examples
```julia
v = RequiredContentValidator(required=["fever", "cough"], mode=:all)
validate(v, "Patient presents with high fever and cough.", nothing) # passed
```
"""
struct RequiredContentValidator <: AbstractValidator
    required::Vector{String}
    forbidden::Vector{String}
    mode::Symbol
    case_sensitive::Bool

    function RequiredContentValidator(;
        required=String[],
        forbidden=String[],
        mode::Symbol=:all,
        case_sensitive::Bool=false
    )
        if mode != :all && mode != :any
            throw(ArgumentError("mode must be :all or :any, got :$mode"))
        end
        req = String[String(x) for x in required]
        forb = String[String(x) for x in forbidden]
        new(req, forb, mode, case_sensitive)
    end
end

function validate(v::RequiredContentValidator, actual, expected=nothing)::ValidationResult
    if !(actual isa AbstractString)
        return ValidationResult(false, "RequiredContentValidator requires string actual output, got $(typeof(actual))"; actual=actual)
    end

    # Determine required items: prefer expected if provided, otherwise v.required
    req_items = if expected !== nothing
        if expected isa AbstractVector
            String[String(x) for x in expected]
        elseif expected isa AbstractString
            String[String(expected)]
        else
            String[string(expected)]
        end
    else
        v.required
    end

    act_text = v.case_sensitive ? String(actual) : lowercase(String(actual))

    found_req = String[]
    missing_req = String[]

    for item in req_items
        match_item = v.case_sensitive ? item : lowercase(item)
        if occursin(match_item, act_text)
            push!(found_req, item)
        else
            push!(missing_req, item)
        end
    end

    # Check forbidden items
    found_forbidden = String[]
    for item in v.forbidden
        match_item = v.case_sensitive ? item : lowercase(item)
        if occursin(match_item, act_text)
            push!(found_forbidden, item)
        end
    end

    # Determine pass/fail based on mode
    req_passed = if isempty(req_items)
        true
    elseif v.mode == :all
        isempty(missing_req)
    elseif v.mode == :any
        !isempty(found_req)
    end

    forbidden_passed = isempty(found_forbidden)
    overall_passed = req_passed && forbidden_passed

    details = Dict{Symbol,Any}(
        :found_required => found_req,
        :missing_required => missing_req,
        :found_forbidden => found_forbidden,
        :mode => v.mode
    )

    if overall_passed
        msg = "Content validation passed. Found: $(join(found_req, ", "))"
        return ValidationResult(true, msg; expected=req_items, actual=actual, details=details)
    else
        reasons = String[]
        if !req_passed
            if v.mode == :all
                push!(reasons, "Missing required items: $(join(missing_req, ", "))")
            else
                push!(reasons, "None of the required items were found: $(join(req_items, ", "))")
            end
        end
        if !forbidden_passed
            push!(reasons, "Found forbidden items: $(join(found_forbidden, ", "))")
        end
        msg = "Content validation failed. " * join(reasons, "; ")
        return ValidationResult(false, msg; expected=req_items, actual=actual, details=details)
    end
end
