#!/usr/bin/env julia
using DrWatson
@quickactivate "HealthLLM"
using HealthLLM
using JSON3
using Dates

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

    if mode in ("zeroshot", "both")
        println("=" ^ 60)
        println("Zero-Shot Benchmark (no RAG)")
        println("Model: $model")
        println("Samples: $sample_limit")
        println("Has DuckDB: $has_db")
        println("=" ^ 60)
        result = HealthLLM.Benchmark.run_zeroshot(model, funsql_path; synthea_kw..., sample_limit=sample_limit)
        println("\nMetrics:")
        println(JSON3.write(result["metrics"], 2))
        println("Failures: $(length(result["failures"]))")
        out = joinpath(@__DIR__, "..", "benchmark_zeroshot_$(Dates.format(now(), "yyyy-mm-dd-HHMMSS")).json")
        open(out, "w") do io; JSON3.write(io, result); end
        println("Results saved to: $out")
    end

    if mode in ("rag", "both")
        println("\n" ^ 60)
        println("RAG Benchmark")
        println("Model: $model")
        println("Embedding: $embedding")
        println("Samples: $sample_limit")
        println("Has DuckDB: $has_db")
        println("=" ^ 60)
        result = HealthLLM.Benchmark.run_rag(model, embedding, funsql_path; synthea_kw..., sample_limit=sample_limit)
        println("\nMetrics:")
        println(JSON3.write(result["metrics"], 2))
        println("Failures: $(length(result["failures"]))")
        out = joinpath(@__DIR__, "..", "benchmark_rag_$(Dates.format(now(), "yyyy-mm-dd-HHMMSS")).json")
        open(out, "w") do io; JSON3.write(io, result); end
        println("Results saved to: $out")
    end
end

main()
