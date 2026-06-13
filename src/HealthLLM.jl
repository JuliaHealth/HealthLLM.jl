module HealthLLM

using PromptingTools
using RAGTools
using LinearAlgebra
using SparseArrays
using JSON3, Serialization
using Statistics

include("pgvector.jl")
include("database.jl")
include("embedding.jl")
include("huggingface.jl")
include("local.jl")
include("utils.jl")
include("grounding.jl")
include("ingestion.jl")
include("query.jl")
include("pipeline.jl")
include("benchmark.jl")

import .Utils: collect_files_with_extensions, write_combined_file, register_models, load_huggingface_model, HuggingFaceLoadResult
import .Embedding: build_index_rag
import .Database: store_embeddings_pgvector
import .Query: generate_funsql_query, generate_funsql_query_direct
import .Grounding: grounding_dir, grounding_files, register_funsql_template!, register_funsql_template_no_context!
import .Ingestion: build_corpus_index, build_corpus_index_from_files, build_grounding_index
import .Pipeline: setup_grounding_index, answer_question, answer_question_direct, quick_demo
import .Benchmark: run_zeroshot, run_rag, run_comparison, compute_detailed_metrics, compute_retrieval_stats, compute_comparison
import .HuggingFaceLLM: HuggingFaceManagedSchema, configure_hf_token!, strip_hf_prefix
import .LocalLLM: LocalManagedSchema

export collect_files_with_extensions, write_combined_file, generate_funsql_query, generate_funsql_query_direct,
build_index_rag, store_embeddings_pgvector, register_models, load_huggingface_model, HuggingFaceLoadResult,
grounding_dir, grounding_files, register_funsql_template!, register_funsql_template_no_context!,
build_corpus_index, build_corpus_index_from_files, build_grounding_index,
setup_grounding_index, answer_question, answer_question_direct, quick_demo,
run_zeroshot, run_rag, run_comparison, compute_detailed_metrics, compute_retrieval_stats, compute_comparison,
HuggingFaceManagedSchema, configure_hf_token!, strip_hf_prefix,
LocalManagedSchema

end
