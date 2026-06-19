using HealthLLM
using HealthLLM.Utils: collect_files_with_extensions, write_combined_file
using RAGTools
using PromptingTools

@testset "Utils" begin
    @testset "collect_files_with_extensions" begin
        mktempdir() do dir
            mkdir(joinpath(dir, "subdir"))
            touch(joinpath(dir, "alpha.jl"))
            touch(joinpath(dir, "beta.JL"))
            touch(joinpath(dir, "subdir", "nested.jl"))
            touch(joinpath(dir, "readme.md"))
            touch(joinpath(dir, "subdir", "data.csv"))

            result = collect_files_with_extensions(dir, [".jl"])
            @test length(result) == 3
            @test all(endswith.(lowercase.(result), ".jl"))
        end

        mktempdir() do dir
            result = collect_files_with_extensions(dir, [".txt", ".csv"])
            @test isempty(result)
        end
    end

    @testset "write_combined_file" begin
        mktempdir() do dir
            file1 = joinpath(dir, "file1.txt")
            file2 = joinpath(dir, "file2.txt")
            output = joinpath(dir, "combined.txt")

            write(file1, "line1\nline2")
            write(file2, "line3")

            write_combined_file([file1, file2], output)

            content = read(output, String)
            @test contains(content, "# File: $file1")
            @test contains(content, "# File: $file2")
            @test contains(content, "line1")
            @test contains(content, "line3")
        end
    end

    @testset "build_index_rag dispatches correctly" begin
        @test_throws Exception HealthLLM.build_index_rag(
            RAGTools.SimpleIndexer(), ["Hello world"]; embedder_kwargs=(;)
        )
    end

    @testset "get_schema inference" begin
        s1 = HealthLLM.Utils.get_schema("Ollama")
        @test typeof(s1) == typeof(PromptingTools.OllamaSchema())

        if isdefined(PromptingTools, :HuggingFaceSchema)
            s2 = HealthLLM.Utils.get_schema(nothing, "hf:facebook/opt-350m")
            @test typeof(s2) == typeof(PromptingTools.HuggingFaceSchema())
        end
    end

    @testset "register_models sets globals" begin
        HealthLLM.register_models("test-chat-model", "test-emb-model")
        @test PromptingTools.MODEL_CHAT == "test-chat-model"
        @test PromptingTools.MODEL_EMBEDDING == "test-emb-model"
    end

    @testset "load_huggingface_model without HF" begin
        res = HealthLLM.load_huggingface_model("some/fake-model")
        @test res isa HealthLLM.HuggingFaceLoadResult
        @test res.downloaded == false
    end
end
