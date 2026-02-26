module Pgvector

"""
    to_pgvector_literal(v)

Convert a numeric vector into Postgres `pgvector` literal syntax.
"""
to_pgvector_literal(v::AbstractVector{<:Real}) = string("[", join(v, ","), "]")

end
