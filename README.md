# HealthLLM.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaHealth.github.io/HealthLLM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaHealth.github.io/HealthLLM.jl/dev/)
[![Build Status](https://github.com/JuliaHealth/HealthLLM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaHealth/HealthLLM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaHealth/HealthLLM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaHealth/HealthLLM.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

`HealthLLM.jl` is a Julia package for retrieval-augmented workflows over health-oriented corpora and structured clinical data. It focuses on a small set of building blocks for preparing corpora, building RAG indexes, generating query-oriented answers, and storing embeddings in PostgreSQL with `pgvector`.

## What the project contains

- Ingestion of curated docs and web-search results, with content-aware chunking that keeps records, schema tables and code blocks whole.
- Embeddings through Ollama or HuggingFace, stored locally, in PostgreSQL/`pgvector`, or in FAISS.
- Grounded prompt construction and natural-language → FunSQL generation, with every generated query parsed, built and rendered against the database schema before it runs.
- A benchmark of 180 OMOP tasks that scores zero-shot against RAG generation by executing the queries.

## Try it

`scripts/end_to_end.jl` takes one question through the whole pipeline (retrieval, prompt, FunSQL, schema check, execution) on the Synthea OMOP database. See the [End-to-End Example](https://juliahealth.github.io/HealthLLM.jl/dev/end-to-end/) for setup.

Detailed setup, usage, and testing instructions live in the hosted docs:
https://juliahealth.github.io/HealthLLM.jl/dev/

## Repository layout

- `src/`: package source code.
- `data/grounding/`: the OMOP schema, FunSQL reference and SQL → FunSQL patterns the model is grounded on.
- `data/benchmark/`: the FunSQL benchmark tasks and how they were built.
- `scripts/`: the end-to-end walkthrough and the benchmark scripts (`scripts/benchmark/`).
- `docs/`: Documenter site and docs-specific environment.
- `test/`: test suite and test-specific environment.
- `CITATION.bib`: citation metadata.

## Documentation

- API and usage guide: `docs/`
- Hosted docs: https://juliahealth.github.io/HealthLLM.jl/dev/

## Citation

See [`CITATION.bib`](CITATION.bib) for citation details.
