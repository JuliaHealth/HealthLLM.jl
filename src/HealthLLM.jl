module HealthLLM

using PromptingTools
using RAGTools
using LinearAlgebra
using SparseArrays
using JSON3, Serialization
using Statistics

include("utils.jl")
include("database.jl")
include("query.jl")

import .Utils: collect_files_with_extensions, write_combined_file, register_models, load_huggingface_model, HuggingFaceLoadResult, build_index_rag
import .Database: store_embeddings_pgvector, validate_embeddings_inputs
import .Query: generate_funsql_query

export PromptingTools, RAGTools
export collect_files_with_extensions, write_combined_file, generate_funsql_query,
    build_index_rag, store_embeddings_pgvector, validate_embeddings_inputs,
    register_models, load_huggingface_model, HuggingFaceLoadResult

end
