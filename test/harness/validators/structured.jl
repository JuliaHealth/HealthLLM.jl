# JSON & Structured Output Validator

using JSON3

"""
    JSONStructureValidator(;
        required_keys::Vector=String[],
        key_types::Dict=Dict(),
        range_constraints::Dict=Dict(),
        allowed_values::Dict=Dict()
    )

Validates that an LLM-generated response is valid JSON and satisfies structural schema requirements.

# Options
- `required_keys`: Keys that MUST be present in the parsed JSON object.
- `key_types`: Dict mapping key names to expected Julia types (e.g. `Dict("confidence" => Number, "condition" => AbstractString)`).
- `range_constraints`: Dict mapping numeric keys to `(min_val, max_val)` tuples (e.g. `Dict("confidence" => (0.0, 1.0))`).
- `allowed_values`: Dict mapping keys to a vector of allowed values / enums (e.g. `Dict("severity" => ["mild", "moderate", "severe"])`).

# Examples
```julia
v = JSONStructureValidator(
    required_keys=["condition", "confidence"],
    key_types=Dict("confidence" => Number, "condition" => AbstractString),
    range_constraints=Dict("confidence" => (0.0, 1.0))
)
validate(v, "{\"condition\": \"hypertension\", \"confidence\": 0.95}", nothing) # passed
```
"""
struct JSONStructureValidator <: AbstractValidator
    required_keys::Vector{String}
    key_types::Dict{String, Type}
    range_constraints::Dict{String, Tuple{Float64, Float64}}
    allowed_values::Dict{String, Vector{Any}}

    function JSONStructureValidator(;
        required_keys=String[],
        key_types=Dict(),
        range_constraints=Dict(),
        allowed_values=Dict()
    )
        req = String[string(k) for k in required_keys]

        types_map = Dict{String, Type}()
        for (k, t) in key_types
            types_map[string(k)] = t
        end

        range_map = Dict{String, Tuple{Float64, Float64}}()
        for (k, (min_v, max_v)) in range_constraints
            range_map[string(k)] = (Float64(min_v), Float64(max_v))
        end

        allowed_map = Dict{String, Vector{Any}}()
        for (k, vals) in allowed_values
            allowed_map[string(k)] = collect(Any, vals)
        end

        new(req, types_map, range_map, allowed_map)
    end
end

function validate(v::JSONStructureValidator, actual, expected=nothing)::ValidationResult
    # 1. Parse JSON
    parsed = try
        if actual isa AbstractString
            # Clean markdown codeblocks if LLM returned ```json ... ```
            cleaned = strip(String(actual))
            if startswith(cleaned, "```json") && endswith(cleaned, "```")
                cleaned = strip(cleaned[8:end-3])
            elseif startswith(cleaned, "```") && endswith(cleaned, "```")
                cleaned = strip(cleaned[4:end-3])
            end
            JSON3.read(cleaned, Dict{String,Any})
        elseif actual isa AbstractDict
            Dict{String,Any}(string(k) => v for (k, v) in actual)
        else
            return ValidationResult(false, "Cannot parse actual value of type $(typeof(actual)) as JSON"; actual=actual)
        end
    catch err
        return ValidationResult(false, "Invalid JSON: $(err)"; actual=actual)
    end

    errors = String[]

    # 2. Check required keys
    for req_key in v.required_keys
        if !haskey(parsed, req_key)
            push!(errors, "Missing required key: '$req_key'")
        end
    end

    # 3. Check types
    for (k, expected_type) in v.key_types
        if haskey(parsed, k)
            val = parsed[k]
            if !isa(val, expected_type)
                push!(errors, "Key '$k' has type $(typeof(val)), expected $(expected_type)")
            end
        end
    end

    # 4. Check range constraints
    for (k, (min_v, max_v)) in v.range_constraints
        if haskey(parsed, k)
            val = parsed[k]
            if val isa Number
                if val < min_v || val > max_v
                    push!(errors, "Key '$k' value $val is outside allowed range [$min_v, $max_v]")
                end
            else
                push!(errors, "Key '$k' is not a number, cannot check range")
            end
        end
    end

    # 5. Check allowed values (enums)
    for (k, allowed) in v.allowed_values
        if haskey(parsed, k)
            val = parsed[k]
            if !(val in allowed)
                push!(errors, "Key '$k' value '$val' is not in allowed values: $allowed")
            end
        end
    end

    passed = isempty(errors)
    msg = passed ? "JSON structure validation passed" :
                   "JSON structure validation failed:\n - " * join(errors, "\n - ")

    return ValidationResult(
        passed,
        msg;
        expected=expected,
        actual=parsed,
        details=Dict(:errors => errors, :parsed => parsed)
    )
end
