module Database

using LibPQ
using ..Pgvector: to_pgvector_literal

"""
    validate_embeddings_inputs(embeddings, chunks, embedding_dimension)

Validate shape consistency between an embedding matrix and associated text chunks.
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

Create an `embeddings` table (if needed) and insert all chunk embeddings in a transaction.
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