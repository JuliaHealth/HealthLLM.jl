#!/usr/bin/env julia
using DrWatson
@quickactivate "HealthLLM"
using HealthLLM
using JSON3
using Dates

function print_detailed_metrics(metrics::Dict, label::String)
    println("\n" * "=" ^ 60)
    println("$label Metrics")
    println("=" ^ 60)
    println("  Total queries:         $(get(metrics, "total", "N/A"))")
    println("  Correct:               $(get(metrics, "correct", "N/A"))")
    println("  Accuracy:              $(get(metrics, "accuracy", "N/A"))")
    println("  Parse success rate:    $(get(metrics, "parse_success_rate", "N/A"))")
    println("  Execution success rate: $(get(metrics, "execution_success_rate", "N/A"))")
    println("  Total duration (s):    $(get(metrics, "total_duration_seconds", "N/A"))")
    println("  Avg duration/query (s): $(get(metrics, "avg_duration_per_query", "N/A"))")

    groups = get(metrics, "accuracy_by_group", nothing)
    if groups !== nothing && !isempty(groups)
        println("\n  Per-group accuracy:")
        for (g, d) in sort!(collect(groups), by=x->x[2]["total"], rev=true)
            println("    $g: $(d["correct"])/$(d["total"]) = $(d["accuracy"])")
        end
    end

    errors = get(metrics, "error_type_breakdown", nothing)
    if errors !== nothing
        println("\n  Error type breakdown:")
        for (k, v) in errors
            v > 0 && println("    $k: $v")
        end
    end

    retrieval = get(metrics, "retrieval", nothing)
    if retrieval !== nothing
        println("\n  Retrieval stats:")
        println("    Total chunks retrieved:  $(get(retrieval, "total_chunks_retrieved", "N/A"))")
        println("    Avg chunks/query:        $(get(retrieval, "avg_chunks_per_query", "N/A"))")
        println("    Unique sources:          $(get(retrieval, "num_unique_sources", "N/A"))")
        sc = get(retrieval, "score_stats", nothing)
        if sc !== nothing && sc["count"] > 0
            println("    Embedding score range:   $(sc["min"]) - $(sc["max"]) (mean: $(sc["mean"]))")
        end
    end
end

function print_comparison(comparison::Dict)
    comp = get(comparison, "comparison", Dict())
    println("\n" * "=" ^ 60)
    println("Comparison: Zero-Shot vs RAG")
    println("=" ^ 60)

    z = get(comparison, "zero_shot", Dict())
    r = get(comparison, "rag", Dict())
    c = get(comp, "comparison", comp)

    z_acc = get(z, "accuracy", 0.0)
    r_acc = get(r, "accuracy", 0.0)
    println("  Zero-shot accuracy:      $(z_acc)")
    println("  RAG accuracy:            $(r_acc)")
    println("  Accuracy delta (RAG-ZS): $(round(r_acc - z_acc, digits=4))")
    println()
    println("  Both correct:    $(get(c, "both_correct", "N/A"))")
    println("  Both wrong:      $(get(c, "both_wrong", "N/A"))")
    println("  RAG correct only: $(get(c, "rag_correct_only", "N/A"))")
    println("  Zero-shot only:  $(get(c, "zero_shot_correct_only", "N/A"))")

    z_parse = get(z, "parse_success_rate", 0.0)
    r_parse = get(r, "parse_success_rate", 0.0)
    z_exec = get(z, "execution_success_rate", 0.0)
    r_exec = get(r, "execution_success_rate", 0.0)
    println("\n  Parse rate:    ZS=$(z_parse)  RAG=$(r_parse)")
    println("  Exec rate:     ZS=$(z_exec)  RAG=$(r_exec)")
end

function main()
    model = length(ARGS) >= 1 ? ARGS[1] : "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct"
    embedding = length(ARGS) >= 2 ? ARGS[2] : "hf:sentence-transformers/all-mpnet-base-v2"
    sample_limit = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
    mode = length(ARGS) >= 4 ? ARGS[4] : "both"

    funsql_path = joinpath(@__DIR__, "..", "FunSQLQueries", "train.jsonl")
    synthea_path = joinpath(@__DIR__, "..", "synthea_1M_3YR.duckdb")

    gen_schema = HealthLLM.Utils.get_schema(nothing, model)
    emb_schema = HealthLLM.Utils.get_schema(nothing, embedding)
    PromptingTools.register_model!(name=model, schema=gen_schema)
    PromptingTools.register_model!(name=embedding, schema=emb_schema)
    PromptingTools.MODEL_CHAT = model
    PromptingTools.MODEL_EMBEDDING = embedding

    has_db = isfile(synthea_path)
    has_data = isfile(funsql_path)

    if !has_data
        println("ERROR: FunSQL dataset not found at $funsql_path")
        return
    end

    synthea_kw = has_db ? (synthea_db_path=synthea_path,) : NamedTuple()

    timestamp = Dates.format(now(), "yyyy-mm-dd-HHMMSS")

    if mode == "comparison"
        println("=" ^ 60)
        println("Comparison Benchmark: Zero-Shot vs RAG")
        println("Model: $model  |  Embedding: $embedding  |  Samples: $sample_limit")
        println("=" ^ 60)
        result = HealthLLM.Benchmark.run_comparison(model, embedding, funsql_path;
            synthea_kw..., sample_limit=sample_limit)
        print_comparison(result["comparison"])
        print_detailed_metrics(result["comparison"]["zero_shot"], "Zero-Shot")
        print_detailed_metrics(result["comparison"]["rag"], "RAG")
        out = joinpath(@__DIR__, "..", "benchmark_comparison_$(timestamp).json")
        open(out, "w") do io; JSON3.write(io, result["comparison"]); end
        println("\nResults saved to: $out")

    elseif mode in ("zeroshot", "zero")
        println("=" ^ 60)
        println("Zero-Shot Benchmark (no RAG)")
        println("Model: $model  |  Samples: $sample_limit  |  Has DuckDB: $has_db")
        println("=" ^ 60)
        result = HealthLLM.Benchmark.run_zeroshot(model, funsql_path;
            synthea_kw..., sample_limit=sample_limit)
        print_detailed_metrics(result["metrics"], "Zero-Shot")
        out = joinpath(@__DIR__, "..", "benchmark_zeroshot_$(timestamp).json")
        open(out, "w") do io; JSON3.write(io, result); end
        println("\nResults saved to: $out")

    elseif mode in ("rag", "rag_only")
        println("=" ^ 60)
        println("RAG Benchmark")
        println("Model: $model  |  Embedding: $embedding  |  Samples: $sample_limit")
        println("Has DuckDB: $has_db")
        println("=" ^ 60)
        result = HealthLLM.Benchmark.run_rag(model, embedding, funsql_path;
            synthea_kw..., sample_limit=sample_limit)
        print_detailed_metrics(result["metrics"], "RAG")
        out = joinpath(@__DIR__, "..", "benchmark_rag_$(timestamp).json")
        open(out, "w") do io; JSON3.write(io, result); end
        println("\nResults saved to: $out")

    elseif mode == "both"
        println("=" ^ 60)
        println("Running: Zero-Shot -> RAG -> Comparison")
        println("Model: $model  |  Embedding: $embedding  |  Samples: $sample_limit")
        println("=" ^ 60)
        result = HealthLLM.Benchmark.run_comparison(model, embedding, funsql_path;
            synthea_kw..., sample_limit=sample_limit)
        print_comparison(result["comparison"])
        out = joinpath(@__DIR__, "..", "benchmark_comparison_$(timestamp).json")
        open(out, "w") do io; JSON3.write(io, result); end
        println("\nResults saved to: $out")

    else
        println("Unknown mode: $mode")
        println("Available modes: zeroshot, rag, both, comparison")
    end
end

main()
