using Test
using HealthLLM

if !@isdefined(TestHarness)
    include(normpath(joinpath(@__DIR__, "..", "harness", "TestHarness.jl")))
    using .TestHarness
end

@testset "Utils Deterministic Unit Tests" begin
    @testset "collect_files_with_extensions" begin
        mktempdir() do tmpdir
            # Create synthetic directory structure
            corpus_files = create_test_corpus(tmpdir)

            # Test collecting only .txt files
            txt_files = HealthLLM.collect_files_with_extensions(tmpdir, [".txt"])
            @test length(txt_files) == 2
            @test all(endswith(f, ".txt") for f in txt_files)

            # Test collecting .txt and .md files
            all_doc_files = HealthLLM.collect_files_with_extensions(tmpdir, [".txt", ".md"])
            @test length(all_doc_files) == 3
            @test any(endswith(f, "hypertension_notes.txt") for f in all_doc_files)
            @test any(endswith(f, "cad_guidelines.md") for f in all_doc_files)
            @test any(endswith(f, "diabetes_summary.txt") for f in all_doc_files)

            # Test non-matching extensions (should ignore .csv)
            jl_files = HealthLLM.collect_files_with_extensions(tmpdir, [".jl"])
            @test isempty(jl_files)

            # Exact validator check
            @test_valid ExactValidator() length(txt_files) 2
        end
    end

    @testset "write_combined_file" begin
        mktempdir() do tmpdir
            f1 = joinpath(tmpdir, "note1.txt")
            f2 = joinpath(tmpdir, "note2.txt")
            out_file = joinpath(tmpdir, "combined.txt")

            write(f1, "Note 1: Normal sinus rhythm.")
            write(f2, "Note 2: Mild bradycardia observed.")

            res_path = HealthLLM.write_combined_file([f1, f2], out_file)
            @test isfile(res_path)
            @test res_path == out_file

            content = read(out_file, String)

            # Validate header format and content concatenation
            @test occursin("# File: $f1", content)
            @test occursin("# File: $f2", content)
            @test occursin("Normal sinus rhythm.", content)
            @test occursin("Mild bradycardia observed.", content)

            # Use harness validators
            @test_valid RequiredContentValidator(required=["Normal sinus rhythm", "Mild bradycardia"]) content
        end
    end
end
