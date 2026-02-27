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

## Minimal Example

```julia
using HealthLLM

files = collect_files_with_extensions("data", [".md", ".jl"])
combined = write_combined_file(files, "combined.txt")
```

```@autodocs
Modules = [HealthLLM]
```