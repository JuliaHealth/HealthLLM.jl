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

## Complete Usage Guide

This guide walks through the full HealthLLM pipeline — from collecting source files to using RAG-augmented responses for database queries.

### 1. Collect source files and build a corpus

Use `collect_files_with_extensions` to recursively gather files by extension, then concatenate them with `write_combined_file`:

```julia
using HealthLLM

# Collect all Julia and Markdown files under "data/"
files = collect_files_with_extensions("data", [".jl", ".md"])

# Write a single combined corpus file with # File: headers
corpus = write_combined_file(files, "corpus.txt")
```

### 2. Register models

Register chat and embedding models for use with PromptingTools. Supports Ollama, HuggingFace, and other backends:

```julia
# Ollama models
register_models("llama3.2", "nomic-embed-text")

# HuggingFace models (auto-detects HuggingFaceSchema)
register_models("hf:facebook/opt-350m", "hf:sentence-transformers/all-mpnet-base-v2")
```

### 3. Build embeddings (RAG index)

Build a RAG index from your source files. The index is used for retrieval-augmented generation:

```julia
using RAGTools

# Build an index with the default simple indexer
index = build_index_rag(RAGTools.SimpleIndexer(), files)
```

### 4. Validate embeddings

Before storing embeddings, validate their shape consistency. The embedding matrix should have rows equal to the embedding dimension and columns equal to the number of chunks:

```julia
# Generate or load embeddings (384-d for nomic-embed-text)
embedding_dim = 384
num_chunks = length(files)
embeddings = rand(embedding_dim, num_chunks)

# Validate dimensions match
validate_embeddings_inputs(embeddings, files, embedding_dim)
# Passes — throws DimensionMismatch if shapes are inconsistent
```

### 5. Store embeddings in PostgreSQL (pgvector)

Store chunk embeddings in a PostgreSQL database with the `pgvector` extension. The function creates an `embeddings` table, validates inputs, and inserts all embeddings in a single transaction:

```julia
using LibPQ

conn = LibPQ.Connection("postgresql://user:pass@localhost:5432/healthdb")

store_embeddings_pgvector(conn, embeddings, files, embedding_dim)
# Creates table:
#   embeddings (id SERIAL, chunk TEXT, embedding VECTOR(384))
```

### 6. Query the RAG system

Retrieve relevant context and generate an answer using the RAG index. The function supports flexible schema detection for both Ollama and HuggingFace backends:

```julia
answer = generate_funsql_query(
    index,
    "nomic-embed-text",          # embedding model for retrieval
    "llama3.2",                  # chat model for generation
    "Context: {input_query}. Answer concisely.",
    "What is the recommended treatment for hypertension?"
)
println(answer)
```

### 7. Use RAG response for a database query

Combine the RAG-generated answer with FunSQL to construct structured database queries. This enables natural-language-driven data exploration:

```julia
using FunSQL
using DuckDB
using DataFrames

# Example: Convert the RAG response into a FunSQL query
# (assuming answer contains a structured query description)
if occursin("hypertension", answer)
    query = FunSQL.From(:patients) |>
            FunSQL.Where(FunSQL.Is(FunSQL.Ref(:diagnosis), "hypertension")) |>
            FunSQL.Select(FunSQL.Ref(:patient_id), FunSQL.Ref(:treatment))
end

# Execute against a DuckDB or PostgreSQL database
conn = DuckDB.DB("health.duckdb")
result = DuckDB.execute(conn, FunSQL.render(query, dialect=FunSQL.SQLDialect(:duckdb))) |> DataFrame
```

```@autodocs
Modules = [HealthLLM, HealthLLM.Utils, HealthLLM.Database, HealthLLM.Query]
```
