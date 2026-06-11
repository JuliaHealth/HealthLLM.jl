using Test
using DrWatson
@quickactivate "HealthLLM"

using HealthLLM

@testset "Benchmark metric utility functions" begin
    # Test load_jsonl
    jsonl_path = joinpath(@__DIR__, "..", "FunSQLQueries", "train.jsonl")
    if isfile(jsonl_path)
        data = HealthLLM.Benchmark.load_jsonl(jsonl_path)
        @test length(data) > 0
        @test haskey(data[1], "query")
        @test haskey(data[1], "sql_query")
        @test haskey(data[1], "response")
    end

    # Test extract_code
    @test HealthLLM.Benchmark.extract_code("```julia\nFrom(:foo)\n```") == "From(:foo)"
    @test HealthLLM.Benchmark.extract_code("```\nFrom(:bar)\n```") == "From(:bar)"
    @test HealthLLM.Benchmark.extract_code("From(:baz)") == "From(:baz)"
    @test HealthLLM.Benchmark.extract_code("Some text ```julia\nx = 1\n``` more") == "x = 1"

    # Test df_equal
    df1 = DataFrame(a=[1, 2, 3], b=[1.0, 2.0, 3.0])
    df2 = DataFrame(a=[1, 2, 3], b=[1.0, 2.0, 3.0])
    @test HealthLLM.Benchmark.df_equal(df1, df2) == true

    df3 = DataFrame(a=[1, 2, 3], b=[1.0, 2.0, 4.0])
    @test HealthLLM.Benchmark.df_equal(df1, df3) == false

    # Reordered columns
    df4 = DataFrame(b=[1.0, 2.0, 3.0], a=[1, 2, 3])
    @test HealthLLM.Benchmark.df_equal(df1, df4) == true

    # Mismatched rows
    df5 = DataFrame(a=[1, 2], b=[1.0, 2.0])
    @test HealthLLM.Benchmark.df_equal(df1, df5) == false

    # Numeric tolerance
    df6 = DataFrame(a=[1, 2, 3], b=[1.0 + 1e-10, 2.0, 3.0])
    @test HealthLLM.Benchmark.df_equal(df1, df6) == true

    df7 = DataFrame(a=[1, 2, 3], b=[1.0 + 1e-4, 2.0, 3.0])
    @test HealthLLM.Benchmark.df_equal(df1, df7; atol=1e-9, rtol=1e-9) == false
end

@testset "Benchmark detailed metrics" begin
    examples = [
        Dict("ok"=>true, "parse_ok"=>true, "exec_ok"=>true, "group"=>"condition_era",
             "error"=>nothing, "duration_seconds"=>1.0),
        Dict("ok"=>true, "parse_ok"=>true, "exec_ok"=>true, "group"=>"condition_era",
             "error"=>nothing, "duration_seconds"=>2.0),
        Dict("ok"=>false, "parse_ok"=>false, "exec_ok"=>false, "group"=>"care_site",
             "error"=>"parse error", "duration_seconds"=>0.5),
        Dict("ok"=>false, "parse_ok"=>true, "exec_ok"=>false, "group"=>"care_site",
             "error"=>"BinderError: x not defined", "duration_seconds"=>0.8),
        Dict("ok"=>false, "parse_ok"=>true, "exec_ok"=>true, "group"=>"drug_era",
             "error"=>nothing, "duration_seconds"=>1.2),
    ]

    metrics = HealthLLM.Benchmark.compute_detailed_metrics(examples)

    @test metrics["total"] == 5
    @test metrics["correct"] == 2
    @test metrics["accuracy"] == 0.4
    @test metrics["parse_success_rate"] == 0.8
    @test metrics["execution_success_rate"] == 0.6

    groups = metrics["accuracy_by_group"]
    @test haskey(groups, "condition_era")
    @test haskey(groups, "care_site")
    @test groups["condition_era"]["accuracy"] == 1.0
    @test groups["care_site"]["accuracy"] == 0.0

    errors = metrics["error_type_breakdown"]
    @test errors["parse_error"] == 1
    @test errors["execution_error"] == 1
    @test errors["result_mismatch"] == 1
end

@testset "Benchmark retrieval stats" begin
    examples = [
        Dict("sources"=>["a.md", "b.md"], "candidate_scores"=>[0.85, 0.72]),
        Dict("sources"=>["a.md"], "candidate_scores"=>[0.91]),
        Dict("sources"=>nothing, "candidate_scores"=>nothing),
        Dict("sources"=>String[], "candidate_scores"=>Float64[]),
    ]

    stats = HealthLLM.Benchmark.compute_retrieval_stats(examples)

    @test stats["total_chunks_retrieved"] == 3
    @test stats["avg_chunks_per_query"] == 0.75
    @test stats["num_unique_sources"] == 2
    @test stats["score_stats"]["count"] == 3
    @test stats["score_stats"]["min"] == 0.72
    @test stats["score_stats"]["max"] == 0.91
end

@testset "Benchmark comparison" begin
    zero_examples = [
        Dict("index"=>1, "nl"=>"q1", "ok"=>true, "group"=>"a", "parse_ok"=>true,
             "exec_ok"=>true, "error"=>nothing, "duration_seconds"=>1.0),
        Dict("index"=>2, "nl"=>"q2", "ok"=>false, "group"=>"a", "parse_ok"=>false,
             "exec_ok"=>false, "error"=>"parse error", "duration_seconds"=>0.5),
        Dict("index"=>3, "nl"=>"q3", "ok"=>true, "group"=>"b", "parse_ok"=>true,
             "exec_ok"=>true, "error"=>nothing, "duration_seconds"=>2.0),
    ]
    rag_examples = [
        Dict("index"=>1, "nl"=>"q1", "ok"=>true, "group"=>"a", "parse_ok"=>true,
             "exec_ok"=>true, "error"=>nothing, "duration_seconds"=>1.5,
             "sources"=>["a.md"], "candidate_scores"=>[0.9]),
        Dict("index"=>2, "nl"=>"q2", "ok"=>true, "group"=>"a", "parse_ok"=>true,
             "exec_ok"=>true, "error"=>nothing, "duration_seconds"=>2.0,
             "sources"=>["a.md"], "candidate_scores"=>[0.85]),
        Dict("index"=>3, "nl"=>"q3", "ok"=>false, "group"=>"b", "parse_ok"=>true,
             "exec_ok"=>true, "error"=>nothing, "duration_seconds"=>3.0,
             "sources"=>["b.md"], "candidate_scores"=>[0.95]),
    ]

    comp = HealthLLM.Benchmark.compute_comparison(zero_examples, rag_examples)

    @test comp["zero_shot"]["accuracy"] ≈ 2/3 atol=0.001
    @test comp["rag"]["accuracy"] ≈ 2/3 atol=0.001
    @test comp["comparison"]["both_correct"] == 1  # q1
    @test comp["comparison"]["both_wrong"] == 0    # none wrong in both
    @test comp["comparison"]["rag_correct_only"] == 1     # q2
    @test comp["comparison"]["zero_shot_correct_only"] == 1  # q3
    @test comp["comparison"]["total_compared"] == 3

    @test haskey(comp["rag"]["retrieval"], "total_chunks_retrieved")
end