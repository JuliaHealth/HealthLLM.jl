```@meta
CurrentModule = HealthLLM
```

# HealthLLM

HealthLLM provides a compact Julia interface for retrieval-augmented workflows over health-related text and structured datasets.

## Package scope

The package centers on five areas:

- collecting source files and writing combined corpora
- ingesting curated docs and web-search results into an index (see [Document Ingestion](ingestion.md))
- building retrieval indexes through `RAGTools`
- constructing grounded prompts from retrieved chunks (see [Querying the RAG System](querying.md))
- generating retrieval-backed answers for query construction
- building and validating embeddings across Ollama and HuggingFace (see [Building Embeddings](embeddings.md))
- storing embeddings in a local file, PostgreSQL/`pgvector`, or FAISS

## Package surface

After `using HealthLLM`, the package API and its RAG dependencies are available from one entrypoint:

```julia
using HealthLLM

register_models("llama3.2", "nomic-embed-text")
files = collect_files_with_extensions("data", [".md", ".jl"])
index = build_index_rag(RAGTools.SimpleIndexer(), files)
```

More detailed setup, testing commands, and the end-to-end walkthrough are in [Getting Started](getting-started.md).

```@autodocs
Modules = [HealthLLM, HealthLLM.Utils, HealthLLM.Database, HealthLLM.Prompt, HealthLLM.Execution, HealthLLM.Query, HealthLLM.Ingestion, HealthLLM.Embeddings, HealthLLM.Storage]
```
