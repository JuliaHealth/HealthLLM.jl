module HealthLLM

using PromptingTools
using RAGTools
using LinearAlgebra
using SparseArrays
using JSON3, Serialization
using Statistics

include("utils.jl")
include("pgvector.jl")
include("database.jl")
include("embedding.jl")
include("query.jl")
include("benchmark.jl")

import .Utils: collect_files_with_extensions, write_combined_file, register_models, register_progress!, clear_progress!
import .Embedding: build_index_rag
import .Database: store_embeddings_pgvector
import .Query: generate_funsql_query
import .Benchmark: run_benchmark

export collect_files_with_extensions, write_combined_file, generate_funsql_query,
build_index_rag, store_embeddings_pgvector, register_models, register_progress!, clear_progress!
export run_benchmark

end
