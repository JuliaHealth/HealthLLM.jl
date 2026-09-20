using Test
using HealthLLM
using PromptingTools

if !@isdefined(TestHarness)
    include(normpath(joinpath(@__DIR__, "..", "harness", "TestHarness.jl")))
    using .TestHarness
end

@testset "Schema & Model Registration Unit Tests" begin
    @testset "get_schema inference" begin
        # HuggingFace-like strings
        hf_schema = HealthLLM.Utils.get_schema(nothing, "hf:facebook/opt-350m")
        if isdefined(PromptingTools, :HuggingFaceSchema)
            @test typeof(hf_schema) == typeof(PromptingTools.HuggingFaceSchema())
        else
            @test typeof(hf_schema) == typeof(PromptingTools.OllamaSchema())
        end

        # Default fallback
        fallback_schema = HealthLLM.Utils.get_schema(nothing, "custom-local-model")
        @test typeof(fallback_schema) == typeof(PromptingTools.OllamaSchema())

        # Explicit schema by name
        if isdefined(PromptingTools, :OpenAISchema)
            openai_schema = HealthLLM.Utils.get_schema("OpenAI", nothing)
            @test typeof(openai_schema) == typeof(PromptingTools.OpenAISchema())
        end
    end

    @testset "register_models" begin
        model_name = "test-chat-model"
        embed_name = "test-embed-model"
        HealthLLM.register_models(model_name, embed_name)

        @test_valid ExactValidator() PromptingTools.MODEL_CHAT model_name
        @test_valid ExactValidator() PromptingTools.MODEL_EMBEDDING embed_name
    end

    @testset "load_huggingface_model deterministic fallback" begin
        res = HealthLLM.load_huggingface_model("mock/medical-model")
        @test isa(res, HealthLLM.HuggingFaceLoadResult)
        @test res.downloaded == false
        @test res.path === nothing
        @test res.info == "mock/medical-model"
    end
end
