module HealthLLM

using PromptingTools
using RAGTools
using LinearAlgebra
using SparseArrays
using JSON3, Serialization
using Statistics

include("utils.jl")
include("database.jl")
include("embeddings.jl")
include("storage.jl")
include("query.jl")
include("ingestion.jl")

import .Utils: collect_files_with_extensions, write_combined_file, register_models, load_huggingface_model, HuggingFaceLoadResult, build_index_rag
import .Database: store_embeddings_pgvector, search_embeddings_pgvector, validate_embeddings_inputs
import .Embeddings: EmbeddingModel, EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL,
    embedding_model, embedding_ref, embedding_dimension,
    embed, cosine_similarity, similarity_matrix,
    validate_embeddings, embedding_sanity_check
import .Storage: AbstractVectorStore, LocalVectorStore, PgVectorStore, FaissVectorStore,
    add!, search, save, load
import .Query: generate_funsql_query
import .Ingestion: SourceDocument, SearchResult,
    AbstractSearchProvider, DuckDuckGoProvider,
    default_search_provider, web_search,
    CURATED_SOURCES, fetch_url, html_to_text, fetch_curated,
    ingest, ingest_to_index

export PromptingTools, RAGTools
export collect_files_with_extensions, write_combined_file, generate_funsql_query,
    build_index_rag, store_embeddings_pgvector, search_embeddings_pgvector,
    validate_embeddings_inputs,
    register_models, load_huggingface_model, HuggingFaceLoadResult
export EmbeddingModel, EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL,
    embedding_model, embedding_ref, embedding_dimension,
    embed, cosine_similarity, similarity_matrix,
    validate_embeddings, embedding_sanity_check
export AbstractVectorStore, LocalVectorStore, PgVectorStore, FaissVectorStore,
    add!, search, save, load
export SourceDocument, SearchResult,
    AbstractSearchProvider, DuckDuckGoProvider,
    default_search_provider, web_search,
    CURATED_SOURCES, fetch_url, html_to_text, fetch_curated,
    ingest, ingest_to_index

end
