using Test
using HealthLLM

@testset "Providers" begin
    # Test provider creation
    hf_provider = HuggingFaceProvider(api_key="test_key")
    @test hf_provider.api_key == "test_key"
    @test hf_provider.model == "Qwen/Qwen3.5-0.8B"

    groq_provider = GroqProvider(api_key="test_key")
    @test groq_provider.api_key == "test_key"

    ollama_provider = OllamaProvider()
    @test ollama_provider.model == "qwen2.5:1.5b"

    gemini_provider = GeminiProvider(api_key="test_key")
    @test gemini_provider.api_key == "test_key"

    openai_provider = OpenAIProvider(api_key="test_key")
    @test openai_provider.api_key == "test_key"

    anthropic_provider = AnthropicProvider(api_key="test_key")
    @test anthropic_provider.api_key == "test_key"

    hf_embedder = HuggingFaceEmbedder(api_key="test_key")
    @test hf_embedder.api_key == "test_key"
end

@testset "Registry" begin
    # Test registry
    register_provider("test_hf", HuggingFaceProvider())
    @test get_provider("test_hf") isa HuggingFaceProvider

    providers = list_providers()
    @test "test_hf" in providers[1]
end