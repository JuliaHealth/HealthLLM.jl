# HealthLLM

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ParamThakkar123.github.io/HealthLLM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ParamThakkar123.github.io/HealthLLM.jl/dev/)
[![Build Status](https://github.com/ParamThakkar123/HealthLLM.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ParamThakkar123/HealthLLM.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/ParamThakkar123/HealthLLM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ParamThakkar123/HealthLLM.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

## Features

HealthLLM.jl supports multiple model APIs for text generation and embeddings, including:

- **HuggingFace**: Use models from HuggingFace via their API.
- **Groq**: Fast inference with Groq's API.
- **Gemini**: Google's Gemini models.
- **OpenAI**: GPT models from OpenAI.
- **Anthropic**: Claude models from Anthropic.
- **Ollama**: Local models via Ollama.

## Usage

### Setting up Providers

Set your API keys in `.env` file or environment variables:

```bash
HUGGINGFACE_TOKEN=your_hf_token
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Using Providers

```julia
using HealthLLM

# Create a provider
provider = OpenAIProvider()

# Generate text
response = generate(provider, "Hello, world!")

# Use in registry
register_provider("my_openai", provider)
```

### Building RAG with Custom Embedders

```julia
embedder = HuggingFaceEmbedder()
index = build_index_rag(cfg, files; embedder=embedder)
```

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).
