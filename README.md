# HealthLLM

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ParamThakkar123.github.io/HealthLLM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ParamThakkar123.github.io/HealthLLM.jl/dev/)
[![Build Status](https://github.com/ParamThakkar123/HealthLLM.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ParamThakkar123/HealthLLM.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/ParamThakkar123/HealthLLM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ParamThakkar123/HealthLLM.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

HealthLLM provides utilities for retrieval-augmented workflows around health data:

- collecting and combining source files for indexing
- building RAG indexes
- generating FunSQL-oriented responses via RAG
- storing embeddings in PostgreSQL `pgvector`

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ParamThakkar123/HealthLLM.jl")
```

## Quick Start

```julia
using HealthLLM

files = collect_files_with_extensions("data", [".md", ".jl"])
combined = write_combined_file(files, "combined.txt")

# Build index and query (configure models/schemas for your provider)
# idx = build_index_rag(cfg, files; embedder_kwargs=(model="...",))
# answer = generate_funsql_query(idx, "embedding-model", "chat-model", "{input_query}", "How many patients?")
```

## Testing

- Default test run executes offline unit tests only:
  - `julia --project=. -e 'using Pkg; Pkg.test()'`
- Integration tests (network + datasets) are opt-in:
  - `HEALTHLLM_RUN_INTEGRATION_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'`

## Environment

- Keep secrets out of git-tracked files.
- Use `.env.example` as a template for local env setup.
- For demo scripts, set `GOOGLE_API_KEY` in your environment.

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).