module Database

using LibPQ
using ..Utils: check_dims, check_k

export store_embeddings_pgvector, search_embeddings_pgvector, validate_embeddings_inputs

function _vector_to_pgarray(v::AbstractVector{T}) where T<:Real
    string("[", join(v, ","), "]")
end

# pgvector distance operators: cosine (`<=>`), negative inner product (`<#>`),
# and L2 (`<->`). All order ascending = most similar first.
function _pg_metric_op(metric::Symbol)
    metric === :cosine && return "<=>"
    metric === :dot && return "<#>"
    metric === :l2 && return "<->"
    throw(ArgumentError("metric must be :cosine, :dot, or :l2, got :$metric"))
end

# Guard identifiers that are interpolated into SQL (table names cannot be bound
# as parameters). Only plain [A-Za-z_][A-Za-z0-9_]* names are allowed.
function _check_identifier(name::AbstractString)
    occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", name) ||
        throw(ArgumentError("Unsafe SQL identifier: $(repr(name))"))
    return name
end

_pg_create_sql(table::AbstractString, dim::Integer) = """
    CREATE TABLE IF NOT EXISTS $(_check_identifier(table)) (
        id SERIAL PRIMARY KEY,
        chunk TEXT NOT NULL,
        embedding VECTOR($dim)
    )
"""

function _pg_search_sql(table::AbstractString, metric::Symbol)
    op = _pg_metric_op(metric)
    return """
        SELECT id, chunk, embedding $op \$1 AS distance
        FROM $(_check_identifier(table))
        ORDER BY embedding $op \$1
        LIMIT \$2
    """
end

"""
    validate_embeddings_inputs(embeddings, chunks, embedding_dimension)

Validate shape consistency between an embedding matrix and associated text chunks.

# Arguments
- `embeddings::AbstractMatrix`: Matrix where rows are embedding dimensions and columns are chunks.
- `chunks::AbstractVector`: Vector of text chunks corresponding to embedding columns.
- `embedding_dimension::Integer`: Expected embedding dimension.

# Returns
`nothing` if validation passes.

# Throws
- `DimensionMismatch`: If embedding dimensions or chunk count don't match.

# Example

```julia
embeddings = rand(384, 10)
chunks = ["chunk \$i" for i in 1:10]
validate_embeddings_inputs(embeddings, chunks, 384)  # passes
```
"""
function validate_embeddings_inputs(
    embeddings::AbstractMatrix,
    chunks::AbstractVector,
    embedding_dimension::Integer
)
    check_dims(embeddings, chunks, embedding_dimension)
    return nothing
end

"""
    store_embeddings_pgvector(conn, embeddings, chunks, embedding_dimension)

Create an `embeddings` table in PostgreSQL with pgvector extension and insert all chunk embeddings in a transaction.

# Arguments
- `conn::LibPQ.Connection`: A connection to the PostgreSQL database.
- `embeddings::AbstractMatrix`: Matrix where rows are embedding dimensions and columns are chunks.
- `chunks::AbstractVector`: Vector of text chunks to store with their embeddings.
- `embedding_dimension::Int`: Dimension of the embedding vectors.

# Returns
`nothing`.

# Example

```julia
conn = LibPQ.Connection("postgresql://user:pass@localhost/db")
embeddings = rand(384, 10)
chunks = ["chunk \$i" for i in 1:10]
store_embeddings_pgvector(conn, embeddings, chunks, 384)
```
"""
function store_embeddings_pgvector(
    conn::LibPQ.Connection,
    embeddings::AbstractMatrix,
    chunks::AbstractVector,
    embedding_dimension::Int;
    table::AbstractString="embeddings"
)
    validate_embeddings_inputs(embeddings, chunks, embedding_dimension)

    LibPQ.execute(conn, _pg_create_sql(table, embedding_dimension))

    dense_embeddings = Matrix{Float64}(embeddings)
    chunk_text = String.(chunks)
    insert_sql = "INSERT INTO $(_check_identifier(table)) (chunk, embedding) VALUES (\$1, \$2)"

    LibPQ.execute(conn, "BEGIN")
    try
        for i in axes(dense_embeddings, 2)
            chunk = chunk_text[i]
            embedding = _vector_to_pgarray(@view dense_embeddings[:, i])
            LibPQ.execute(conn, insert_sql, (chunk, embedding))
        end
        LibPQ.execute(conn, "COMMIT")
    catch
        LibPQ.execute(conn, "ROLLBACK")
        rethrow()
    end

    return nothing
end

"""
    search_embeddings_pgvector(conn, query, k; table="embeddings", metric=:cosine)

Return the `k` nearest stored chunks to `query` from a pgvector `table`, ordered
most-similar first. `metric` selects the distance operator: `:cosine` (`<=>`),
`:dot` (`<#>`), or `:l2` (`<->`). Each result is a `(; id, chunk, distance)`
named tuple.

# Example

```julia
hits = search_embeddings_pgvector(conn, rand(384), 5)
for h in hits
    println(h.distance, "  ", first(h.chunk, 60))
end
```
"""
function search_embeddings_pgvector(
    conn::LibPQ.Connection,
    query::AbstractVector,
    k::Integer;
    table::AbstractString="embeddings",
    metric::Symbol=:cosine
)
    check_k(k)
    qvec = _vector_to_pgarray(Vector{Float64}(query))
    res = LibPQ.execute(conn, _pg_search_sql(table, metric), (qvec, k))
    return [(; id=row.id, chunk=row.chunk, distance=row.distance) for row in res]
end

end

