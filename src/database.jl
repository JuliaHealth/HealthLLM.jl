module Database
using LibPQ
using ..Pgvector: to_pgvector_literal

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
chunks = ["chunk $i" for i in 1:10]
validate_embeddings_inputs(embeddings, chunks, 384)  # passes
```
"""
function validate_embeddings_inputs(
    embeddings::AbstractMatrix,
    chunks::AbstractVector,
    embedding_dimension::Integer
)
    n_rows, n_cols = size(embeddings)
    n_rows == embedding_dimension || throw(
        DimensionMismatch(
            "Embedding height ($n_rows) must match embedding_dimension ($embedding_dimension)."
        )
    )
    length(chunks) == n_cols || throw(
        DimensionMismatch(
            "Number of chunks ($(length(chunks))) must match number of embedding columns ($n_cols)."
        )
    )
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
chunks = ["chunk $i" for i in 1:10]
store_embeddings_pgvector(conn, embeddings, chunks, 384)
```
"""
function store_embeddings_pgvector(
    conn::LibPQ.Connection,
    embeddings::AbstractMatrix,
    chunks::AbstractVector,
    embedding_dimension::Int
)
    validate_embeddings_inputs(embeddings, chunks, embedding_dimension)

    LibPQ.execute(conn, """
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            chunk TEXT NOT NULL,
            embedding VECTOR($embedding_dimension)
        )
    """)

    dense_embeddings = Matrix{Float64}(embeddings)
    chunk_text = String.(chunks)

    LibPQ.execute(conn, "BEGIN")
    try
        for i in axes(dense_embeddings, 2)
            chunk = chunk_text[i]
            embedding = to_pgvector_literal(@view dense_embeddings[:, i])
            LibPQ.execute(conn, """
                INSERT INTO embeddings (chunk, embedding)
                VALUES (\$1, \$2)
            """, (chunk, embedding))
        end
        LibPQ.execute(conn, "COMMIT")
    catch
        LibPQ.execute(conn, "ROLLBACK")
        rethrow()
    end

    return nothing
end

end