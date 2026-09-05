using HealthLLM
using Test

# Guards the re-export mechanism in src/HealthLLM.jl: anything a submodule
# exports must be reachable from `HealthLLM` itself. Before this was automated,
# the whole chunking API was exported by `Ingestion` and silently missing from
# the package's own export list.
@testset "Exports" begin
    @testset "every submodule export is re-exported" begin
        top = Set(names(HealthLLM))
        for m in HealthLLM.PUBLIC_MODULES
            for n in names(m)
                n === nameof(m) && continue
                @test n in top
            end
        end
    end

    @testset "representative symbols resolve on HealthLLM" begin
        for n in (:build_index_rag, :get_schema, :render_provenance,      # Utils
            :store_embeddings_pgvector,                                    # Database
            :embed, :cosine_similarity, :DEFAULT_EMBEDDING_MODEL,          # Embeddings
            :LocalVectorStore, :Hit, :add!, :search, :retrieve,            # Storage
            :build_prompt, :FUNSQL_SYSTEM_PROMPT,                          # Prompt
            :generate_funsql, :sanity_check_funsql,                        # Execution
            :generate_funsql_query, :DEFAULT_QUERY_TEMPLATE,               # Query
            :ingest, :ingest_to_index, :fetch_url,                         # Ingestion
            :Chunk, :chunk, :chunk_document, :chunk_provenance,            # Ingestion/chunk
            :RecordChunk, :HeaderChunk, :RecursiveChunk, :FixedSizeChunk,
            :default_strategy, :load_funsql_examples)
            @test isdefined(HealthLLM, n)
            @test n in names(HealthLLM)
        end
    end

    @testset "upstream packages are re-exported" begin
        @test :PromptingTools in names(HealthLLM)
        @test :RAGTools in names(HealthLLM)
    end
end
