
using Test
@quickactivate "HealthLLM"

using HealthLLM
using PromptingTools
using RAGTools

@testset "HuggingFace model support" begin
    if !isdefined(PromptingTools, :HuggingFaceSchema)
        @testskip "PromptingTools does not provide HuggingFaceSchema; skipping HF integration tests."
    end

    # Schema detection should return a HuggingFaceSchema for HF-like model strings
    hf_schema = Utils.get_schema(nothing, "hf:facebook/opt-350m")
    @test typeof(hf_schema) == typeof(PromptingTools.HuggingFaceSchema())

    # Register models should set global model names
    HealthLLM.register_models("hf:facebook/opt-350m", "hf:sentence-transformers/all-mpnet-base-v2")
    @test PromptingTools.MODEL_CHAT == "hf:facebook/opt-350m"
    @test PromptingTools.MODEL_EMBEDDING == "hf:sentence-transformers/all-mpnet-base-v2"

    # Monkeypatch RAGTools.airag to capture kwargs passed by generate_funsql_query
    original_airag = RAGTools.airag
    called = Dict{Symbol,Any}()

    RAGTools.airag = (index; kwargs...) -> begin
        called[:kwargs] = kwargs
        return "dummy-answer"
    end

    try
        ans = HealthLLM.generate_funsql_query(nothing, "hf:sentence-transformers/all-mpnet-base-v2", "hf:facebook/opt-350m", "{input_query}", "hello")
        @test ans == "dummy-answer"

        kwargs = called[:kwargs]
        @test haskey(kwargs, :retriever_kwargs)
        @test haskey(kwargs, :generator_kwargs)

        retriever = kwargs[:retriever_kwargs]
        generator = kwargs[:generator_kwargs]

        @test typeof(retriever[:schema]) == typeof(PromptingTools.HuggingFaceSchema())
        @test typeof(generator[:schema]) == typeof(PromptingTools.HuggingFaceSchema())
    finally
        # restore original method
        RAGTools.airag = original_airag
    end
end
