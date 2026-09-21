using HealthLLM
using HealthLLM.Utils: collect_files_with_extensions, write_combined_file

@testset "Utils" begin
    @testset "re-exported modules are available" begin
        @test isdefined(Main, :RAGTools)
        @test isdefined(Main, :PromptingTools)
        @test RAGTools === HealthLLM.RAGTools
        @test PromptingTools === HealthLLM.PromptingTools
    end

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

        # explicit provider form
        @test typeof(get_schema(:ollama)) == typeof(PromptingTools.OllamaSchema())
        @test get_schema(:huggingface) isa HuggingFaceOpenAISchema
        @test_throws ArgumentError get_schema(:nonsense)

        # schema-name form recognises HuggingFace by name
        @test get_schema("HuggingFace") isa HuggingFaceOpenAISchema
        @test get_schema("huggingface") isa HuggingFaceOpenAISchema

        # heuristic form: HF-looking model names route to the HF schema
        for name in ("hf:facebook/opt-350m", "some-huggingface-model", "hf/x", "hf-x")
            @test get_schema(nothing, name) isa HuggingFaceOpenAISchema
        end

        @test typeof(get_schema(nothing, "llama3.2")) ==
              typeof(PromptingTools.OllamaSchema())
    end

    @testset "register_models sets globals" begin
        HealthLLM.register_models("test-chat-model", "test-emb-model")
        @test PromptingTools.MODEL_CHAT == "test-chat-model"
        @test PromptingTools.MODEL_EMBEDDING == "test-emb-model"

        # HuggingFace-style names register just as well
        HealthLLM.register_models("hf:facebook/opt-350m",
            "hf:sentence-transformers/all-mpnet-base-v2")
        @test PromptingTools.MODEL_CHAT == "hf:facebook/opt-350m"
        @test PromptingTools.MODEL_EMBEDDING == "hf:sentence-transformers/all-mpnet-base-v2"
    end

    @testset "load_huggingface_model reports failure without throwing" begin
        # HuggingFaceHub is a real dependency now, so this reaches the hub (or
        # fails to); either way an unknown repo must come back as not-downloaded
        # rather than raising.
        res = HealthLLM.load_huggingface_model("healthllm-jl/definitely-not-a-real-model")
        @test res isa HealthLLM.HuggingFaceLoadResult
        @test res.downloaded == false
        @test res.path === nothing
    end

    @testset "require_main_module" begin
        @test require_main_module(:Test, "unreachable") === Main.Test
        err = try
            require_main_module(:NotLoadedAnywhere, "install it first")
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("install it first", err.msg)
    end

    @testset "check_dims / check_k" begin
        E = rand(Float32, 4, 3)
        @test check_dims(E) == (4, 3)
        @test check_dims(E, ["a", "b", "c"], 4) == (4, 3)
        @test_throws DimensionMismatch check_dims(E, ["a", "b", "c"], 99)
        @test_throws DimensionMismatch check_dims(E, ["only-one"], 4)

        @test check_k(3) == 3
        @test_throws ArgumentError check_k(0)
        @test_throws ArgumentError check_k(-1)
    end

    @testset "render_provenance" begin
        @test render_provenance(Dict{Symbol,Any}(:url => "http://cdm", :heading => "person")) ==
              "http://cdm › person"
        # source stands in for a missing url; group stands in for a missing heading
        @test render_provenance(Dict{Symbol,Any}(:source => "FunSQL", :group => "joins")) ==
              "FunSQL › joins"
        # either half may be absent
        @test render_provenance(Dict{Symbol,Any}(:url => "http://x")) == "http://x"
        @test render_provenance(Dict{Symbol,Any}(:heading => "person")) == "person"
        @test render_provenance(Dict{Symbol,Any}()) == ""
        @test length(render_provenance(
            Dict{Symbol,Any}(:url => repeat("u", 400), :heading => repeat("h", 400)))) == 512
    end
end
