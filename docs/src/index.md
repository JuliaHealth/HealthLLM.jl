```@meta
CurrentModule = HealthLLM
```

# HealthLLM

Documentation for [HealthLLM](https://github.com/ParamThakkar123/HealthLLM.jl).

## Overview

HealthLLM includes utilities for:

- collecting source files and writing combined corpora
- building RAG indexes
- generating FunSQL-oriented answers with retrieval
- storing embeddings in PostgreSQL `pgvector`

## Example Pipeline

```julia
using HealthLLM
using LibPQ

# 1. Collect source files and write a combined corpus
files = collect_files_with_extensions("data", [".md", ".jl"])
combined = write_combined_file(files, "combined.txt")

# 2. Register models (supports Ollama, HuggingFace, etc.)
register_models("llama3.2", "nomic-embed-text")

# 3. Build a RAG index from the corpus
index = build_index_rag(files)

# 4. Generate embeddings and validate their dimensions
embeddings = rand(384, length(files))  # e.g. nomic-embed-text produces 384-d vectors
embedding_dim = 384
validate_embeddings_inputs(embeddings, files, embedding_dim)

# 5. Store embeddings in PostgreSQL with pgvector
conn = LibPQ.Connection("postgresql://user:pass@localhost/db")
store_embeddings_pgvector(conn, embeddings, files, embedding_dim)

# 6. Query the RAG system
answer = generate_funsql_query(
    index,
    "nomic-embed-text",
    "llama3.2",
    "Context: {input_query}. Answer concisely.",
    "What is the recommended treatment for hypertension?"
)

# 7. Use the RAG response for a database query (via FunSQL or similar)
println(answer)
```

```@autodocs
Modules = [HealthLLM, HealthLLM.Utils]
```
