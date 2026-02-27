using Test

@testset "File Utilities" begin
    mktempdir() do tmpdir
        subdir = joinpath(tmpdir, "nested")
        mkpath(subdir)

        file_jl = joinpath(tmpdir, "a.jl")
        file_md = joinpath(subdir, "b.md")
        file_txt = joinpath(subdir, "ignore.txt")

        write(file_jl, "println(\"hello\")\n")
        write(file_md, "# title\n")
        write(file_txt, "ignored\n")

        files = HealthLLM.collect_files_with_extensions(tmpdir, [".jl", ".md"])
        @test Set(files) == Set([file_jl, file_md])

        output_file = joinpath(tmpdir, "combined.txt")
        returned = HealthLLM.write_combined_file(sort(files), output_file)
        @test returned == output_file

        combined = read(output_file, String)
        @test occursin("# File: $file_jl", combined)
        @test occursin("# File: $file_md", combined)
        @test occursin("println(\"hello\")", combined)
        @test occursin("# title", combined)
    end
end

@testset "pgvector Helpers" begin
    @test HealthLLM.Pgvector.to_pgvector_literal([1, 2, 3]) == "[1,2,3]"
    @test HealthLLM.Pgvector.to_pgvector_literal([1.5, 2.0]) == "[1.5,2.0]"
end

@testset "Embedding Validation" begin
    embeddings = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    chunks = ["chunk-a", "chunk-b"]

    @test isnothing(
        HealthLLM.Database.validate_embeddings_inputs(embeddings, chunks, 3)
    )
    @test_throws DimensionMismatch HealthLLM.Database.validate_embeddings_inputs(
        embeddings, chunks, 4
    )
    @test_throws DimensionMismatch HealthLLM.Database.validate_embeddings_inputs(
        embeddings, ["chunk-a"], 3
    )
end

@testset "Public Exports" begin
    @test :register_models in names(HealthLLM)
end

@testset "Embedding Module Wiring" begin
    err = try
        HealthLLM.build_index_rag(nothing, String[])
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test !(err isa UndefVarError)
end