using Test
using DrWatson
@quickactivate "HealthLLM"

using HealthLLM
using PromptingTools
using RAGTools

@testset "HuggingFace model support" begin
    hf_schema = HealthLLM.Utils.get_schema(nothing, "hf:facebook/opt-350m")
    @test typeof(hf_schema) == typeof(HealthLLM.HuggingFaceManagedSchema())

    HealthLLM.register_models("hf:facebook/opt-350m", "hf:sentence-transformers/all-mpnet-base-v2")
    @test PromptingTools.MODEL_CHAT == "hf:facebook/opt-350m"
    @test PromptingTools.MODEL_EMBEDDING == "hf:sentence-transformers/all-mpnet-base-v2"

    gen_schema = HealthLLM.Utils.get_schema(nothing, "hf:facebook/opt-350m")
    emb_schema = HealthLLM.Utils.get_schema(nothing, "hf:sentence-transformers/all-mpnet-base-v2")
    @test typeof(gen_schema) == typeof(HealthLLM.HuggingFaceManagedSchema())
    @test typeof(emb_schema) == typeof(HealthLLM.HuggingFaceManagedSchema())
end
