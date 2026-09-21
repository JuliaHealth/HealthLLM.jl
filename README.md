# HealthLLM.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaHealth.github.io/HealthLLM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaHealth.github.io/HealthLLM.jl/dev/)
[![Build Status](https://github.com/JuliaHealth/HealthLLM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaHealth/HealthLLM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaHealth/HealthLLM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaHealth/HealthLLM.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

`HealthLLM.jl` is a Julia package for retrieval-augmented workflows over health-oriented corpora and structured clinical data. It focuses on a small set of building blocks for preparing corpora, building RAG indexes, generating query-oriented answers, and storing embeddings in PostgreSQL with `pgvector`.

## What the project contains

- Corpus preparation utilities for collecting files and writing combined sources.
- RAG pipeline helpers built around `RAGTools` and `PromptingTools`.
- Query-generation helpers for retrieval-backed, FunSQL-oriented workflows.
- PostgreSQL embedding storage utilities targeting `pgvector`.

Detailed setup, usage, and testing instructions live in the hosted docs:
https://juliahealth.github.io/HealthLLM.jl/dev/

## Repository layout

- `src/`: package source code.
- `docs/`: Documenter site and docs-specific environment.
- `test/`: test suite and test-specific environment.
- `FunSQLQueries/`: query-related assets used by the project.
- `.env.example`: example environment variable template.
- `CITATION.bib`: citation metadata.

## Documentation

- API and usage guide: `docs/`
- Hosted docs: https://juliahealth.github.io/HealthLLM.jl/dev/

## Citation

See [`CITATION.bib`](CITATION.bib) for citation details.
