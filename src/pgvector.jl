module Pgvector

"""
    to_pgvector_literal(v::AbstractVector{<:Real})

Convert a numeric vector into Postgres `pgvector` literal syntax.

# Arguments
- `v::AbstractVector{<:Real}`: A vector of numerical values to convert.

# Returns
A string representation of `v` in `pgvector` literal format.

# Example

```julia
julia> to_pgvector_literal([1, 2, 3])
"[1,2,3]"
```
"""
to_pgvector_literal(v::AbstractVector{<:Real}) = string("[", join(v, ","), "]")

end
