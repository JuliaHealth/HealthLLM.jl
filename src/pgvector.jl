module Pgvector
    _vector_to_pgarray(v::AbstractVector{T}) where T<:Real = string("[", join(v, ","), "]")
end