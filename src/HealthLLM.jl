module HealthLLM

using PromptingTools
using RAGTools
using LinearAlgebra
using SparseArrays
using JSON3, Serialization

include("utils.jl")
include("providers.jl")
include("registry.jl")
include("rag.jl")
include("pgvector.jl")
include("database.jl")
include("embedding.jl")
include("query.jl")

import .Utils: collect_files_with_extensions, write_combined_file, register_models
import .Providers: ModelProvider, EmbeddingProvider, HuggingFaceProvider, GroqProvider, OllamaProvider, GeminiProvider, OpenAIProvider, AnthropicProvider, HuggingFaceEmbedder, generate
import .Registry: register_provider, get_provider, list_providers
import .RAG: SimpleEmbedder, prepare_data, chunk_text, build_index, ChunkResult, retrieve_chunks, generate_answer
import .Embedding: build_index_rag
import .Database: store_embeddings_pgvector
import .Query: generate_funsql_query

export collect_files_with_extensions, write_combined_file, generate_funsql_query,
build_index_rag, store_embeddings_pgvector,
ModelProvider, EmbeddingProvider, HuggingFaceProvider, GroqProvider, OllamaProvider, GeminiProvider, OpenAIProvider, AnthropicProvider, HuggingFaceEmbedder,
register_provider, get_provider, list_providers, generate,
SimpleEmbedder, prepare_data, chunk_text, build_index, ChunkResult, retrieve_chunks, generate_answer

end