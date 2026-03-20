@testset "Utils" begin
    using HealthLLM.Utils

    @testset "collect_files_with_extensions" begin
        mktempdir() do dir
            mkdir(joinpath(dir, "subdir"))
            touch(joinpath(dir, "file.jl"))
            touch(joinpath(dir, "file.JL"))
            touch(joinpath(dir, "subdir", "nested.jl"))
            touch(joinpath(dir, "readme.md"))
            touch(joinpath(dir, "subdir", "data.csv"))

            result = collect_files_with_extensions(dir, [".jl"])
            @test length(result) == 3
            @test all(endswith(lowercase.(result), ".jl"))
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

    @testset "to_pgvector_literal" begin
        using HealthLLM.Pgvector
        @test to_pgvector_literal([1.0, 2.0, 3.0]) == "[1.0,2.0,3.0]"
        @test to_pgvector_literal([0, 1, 2]) == "[0,1,2]"
        @test to_pgvector_literal(Float32[0.1, 0.2]) == "[0.1,0.2]"
    end
end
