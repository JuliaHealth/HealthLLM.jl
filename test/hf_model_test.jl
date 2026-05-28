
using Test
using DrWatson
@quickactivate "HealthLLM"

using HealthLLM
using PromptingTools
using RAGTools

@testset "HuggingFace model support" begin
    # Schema detection should return a HuggingFaceSchema for HF-like model strings
    hf_schema = HealthLLM.Utils.get_schema(nothing, "hf:facebook/opt-350m")
    if isdefined(PromptingTools, :HuggingFaceSchema)
        @test typeof(hf_schema) == typeof(PromptingTools.HuggingFaceSchema())
    else
        @test typeof(hf_schema) == typeof(PromptingTools.OllamaSchema())
    end

    # Register models should set global model names
    HealthLLM.register_models("hf:facebook/opt-350m", "hf:sentence-transformers/all-mpnet-base-v2")
    @test PromptingTools.MODEL_CHAT == "hf:facebook/opt-350m"
    @test PromptingTools.MODEL_EMBEDDING == "hf:sentence-transformers/all-mpnet-base-v2"

    # Confirm schema inference for generator & embedder without invoking RAG internals
    gen_schema = HealthLLM.Utils.get_schema(nothing, "hf:facebook/opt-350m")
    emb_schema = HealthLLM.Utils.get_schema(nothing, "hf:sentence-transformers/all-mpnet-base-v2")

    if isdefined(PromptingTools, :HuggingFaceSchema)
        @test typeof(gen_schema) == typeof(PromptingTools.HuggingFaceSchema())
        @test typeof(emb_schema) == typeof(PromptingTools.HuggingFaceSchema())
    else
        @test typeof(gen_schema) == typeof(PromptingTools.OllamaSchema())
        @test typeof(emb_schema) == typeof(PromptingTools.OllamaSchema())
    end
end
