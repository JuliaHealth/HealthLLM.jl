using Pkg
Pkg.activate(".")
include("src/HealthLLM.jl")

using LibPQ
using RAGTools

# Assume DB is set up with pgvector
# Connection details
conn = LibPQ.Connection("dbname=healthllm user=postgres password=password host=localhost")

# Load combined data
combined_file = "OHDSI_FunSQL_combined.txt"
text = read(combined_file, String)

# Chunk the text
chunks = RAGTools.chunk_text(text, 1000, 200)  # chunk size 1000, overlap 200

# Build embeddings
cfg = RAGTools.ChunkEmbeddingsConfig(
    embedder = :huggingface,
    model = "sentence-transformers/all-MiniLM-L6-v2"
)
index = HealthLLM.build_index_rag(cfg, [text])  # Wait, build_index takes files, but we have text.

# Actually, RAGTools.build_index takes sources, which can be strings or files.

# To fix, perhaps pass the file.

index = HealthLLM.build_index_rag(cfg, [combined_file])

# Now, to store in DB, but store_embeddings_pgvector takes embeddings matrix and chunks.

# Need to extract from index.

# The index has chunks and embeddings.

# Perhaps modify the store function to take the index.

# For now, let's assume we extract.

embeddings = index.embeddings
chunks_text = [chunk.text for chunk in index.chunks]

# Store
HealthLLM.store_embeddings_pgvector(conn, embeddings, chunks_text, size(embeddings,1))

close(conn)