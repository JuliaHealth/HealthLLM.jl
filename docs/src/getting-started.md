# Getting Started

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ParamThakkar123/HealthLLM.jl")
```

## Examples

The `examples/` directory contains runnable scripts for trying the package surface directly.

- `examples/rag_pipeline.jl`: end-to-end example covering file collection, model registration, index construction, embedding validation, and query generation.

You can run the pipeline example from the repository root with:

```bash
julia --project=. examples/rag_pipeline.jl
```

Users can also treat the example as a starting point for trying individual features of the pipeline, such as corpus preparation, retrieval index construction, or query generation in isolation.

## Quick start

```julia
using HealthLLM

files = collect_files_with_extensions("data", [".md", ".jl"])
corpus = write_combined_file(files, "combined.txt")

register_models("llama3.2", "nomic-embed-text")
index = build_index_rag(RAGTools.SimpleIndexer(), files)

answer = generate_funsql_query(
    index,
    "nomic-embed-text",
    "llama3.2",
    "Context: {input_query}. Answer concisely.",
    "What is the FunSQL query to query for patients with Hypertension?"
)
```

## Full workflow

### 1. Collect source files and build a corpus

Use `collect_files_with_extensions` to recursively gather files by extension, then concatenate them with `write_combined_file`:

```julia
using HealthLLM

files = collect_files_with_extensions("data", [".jl", ".md"])
corpus = write_combined_file(files, "corpus.txt")
```

### 2. Register models

Register chat and embedding models for use with `PromptingTools`. Supports Ollama, HuggingFace, and other backends:

```julia
register_models("llama3.2", "nomic-embed-text")
register_models("hf:facebook/opt-350m", "hf:sentence-transformers/all-mpnet-base-v2")
```

### 3. Build a RAG index

```julia
index = build_index_rag(RAGTools.SimpleIndexer(), files)
```

### 4. Validate embeddings

```julia
embedding_dim = 384
num_chunks = length(files)
embeddings = rand(embedding_dim, num_chunks)

validate_embeddings_inputs(embeddings, files, embedding_dim)
```

### 5. Store embeddings in PostgreSQL

```julia
using LibPQ

conn = LibPQ.Connection("postgresql://user:pass@localhost:5432/healthdb")
store_embeddings_pgvector(conn, embeddings, files, embedding_dim)
```

### 6. Query the RAG system

```julia
answer = generate_funsql_query(
    index,
    "nomic-embed-text",
    "llama3.2",
    "Context: {input_query}. Answer concisely.",
    "What is the recommended treatment for hypertension?"
)
```

## Testing

Default test runs execute the offline unit tests:

```julia
using Pkg
Pkg.test()
```

Integration-style tests that depend on network or large datasets are opt-in:

```bash
HEALTHLLM_RUN_INTEGRATION_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'
```

## Environment notes

- Keep secrets out of git-tracked files.
- Use `.env.example` as a local template.
- Configure provider keys in your environment before running model-backed examples.
