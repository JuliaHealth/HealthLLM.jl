using Test
using HealthLLM

@testset "Utils tests" begin
    @testset "_vector_to_pgarray" begin
        @test HealthLLM.Database._vector_to_pgarray([1.0, 2.0, 3.0]) == "[1.0,2.0,3.0]"
        @test HealthLLM.Database._vector_to_pgarray([1, 2, 3]) == "[1,2,3]"
        @test HealthLLM.Database._vector_to_pgarray(Float64[]) == "[]"
    end

    @testset "build_index_rag dispatches correctly" begin
        @test_throws Exception HealthLLM.build_index_rag(
            RAGTools.SimpleIndexer(), ["Hello world"]; embedder_kwargs=(;)
        )
    end

    @testset "strip_hf_prefix" begin
        @test HealthLLM.strip_hf_prefix("hf:facebook/opt-350m") == "facebook/opt-350m"
        @test HealthLLM.strip_hf_prefix("facebook/opt-350m") == "facebook/opt-350m"
        @test HealthLLM.strip_hf_prefix("hf:") == ""
        @test HealthLLM.strip_hf_prefix("") == ""
    end

    @testset "_strip_prefix" begin
        @test HealthLLM.LocalLLM._strip_prefix("local:facebook/opt-350m") == "facebook/opt-350m"
        @test HealthLLM.LocalLLM._strip_prefix("hf:facebook/opt-350m") == "facebook/opt-350m"
        @test HealthLLM.LocalLLM._strip_prefix("facebook/opt-350m") == "facebook/opt-350m"
    end

    @testset "get_schema variants" begin
        hf_schema = HealthLLM.Utils.get_schema(nothing, "hf:facebook/opt-350m")
        @test typeof(hf_schema) == typeof(HealthLLM.HuggingFaceManagedSchema())

        local_schema = HealthLLM.Utils.get_schema(nothing, "local:facebook/opt-350m")
        @test typeof(local_schema) == typeof(HealthLLM.LocalManagedSchema())

        default_schema = HealthLLM.Utils.get_schema(nothing, "unknown-model")
        @test typeof(default_schema) == typeof(PromptingTools.OllamaSchema())

        explicit_schema = HealthLLM.Utils.get_schema("Ollama", nothing)
        @test typeof(explicit_schema) == typeof(PromptingTools.OllamaSchema())
    end
end

@testset "Grounding tests" begin
    @testset "grounding_dir" begin
        dir = HealthLLM.grounding_dir()
        @test dir !== nothing
        @test isdir(dir)
    end

    @testset "grounding_files" begin
        files = HealthLLM.grounding_files()
        @test length(files) >= 3
        @test all(f -> endswith(f, ".md"), files)
    end

    @testset "register_funsql_templates" begin
        key1 = :TestTemplate1
        key2 = :TestTemplate2
        HealthLLM.register_funsql_template!(name=key1)
        @test haskey(PromptingTools.TEMPLATE_STORE, key1)
        @test PromptingTools.TEMPLATE_STORE[key1] isa Vector

        HealthLLM.register_funsql_template_no_context!(name=key2)
        @test haskey(PromptingTools.TEMPLATE_STORE, key2)
        @test PromptingTools.TEMPLATE_STORE[key2] isa Vector

        ret1 = HealthLLM.register_funsql_template!(name=key1)
        @test ret1 == key1
    end
end
