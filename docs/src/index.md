```@meta
CurrentModule = HealthLLM
```

# HealthLLM

HealthLLM provides a compact Julia interface for retrieval-augmented workflows over health-related text and structured datasets.

## Package scope

The package centers on four areas:

- collecting source files and writing combined corpora
- building retrieval indexes through `RAGTools`
- generating retrieval-backed answers for query construction
- storing embeddings in PostgreSQL with `pgvector`

## Package surface

After `using HealthLLM`, the package API and its RAG dependencies are available from one entrypoint:

```julia
using HealthLLM

register_models("llama3.2", "nomic-embed-text")
files = collect_files_with_extensions("data", [".md", ".jl"])
index = build_index_rag(RAGTools.SimpleIndexer(), files)
```

Users who want a runnable walkthrough can start with `examples/rag_pipeline.jl`. More detailed setup, testing commands, and the end-to-end walkthrough are in [Getting Started](getting-started.md).

```@autodocs
Modules = [HealthLLM, HealthLLM.Utils, HealthLLM.Database, HealthLLM.Query]
```
