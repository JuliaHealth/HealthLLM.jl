#!/usr/bin/env julia
# Run the FunSQL benchmark using HealthLLM against the FunSQL dataset.
# Usage:
#   julia scripts/run_benchmark_llama.jl [model_name] [model_embedding] [sample_limit]
# Example:
#   julia scripts/run_benchmark_llama.jl "meta-llama/Llama-2-7b-chat-hf" "sentence-transformers/all-MiniLM-L6-v2" 20

using HealthLLM
using HuggingFaceHub
const HF = HuggingFaceHub
using JSON3
using Dates
# progress bar (required)
using ProgressMeter
const HAS_PROGRESS = true

function download_datasets(; synthea_path::Union{Nothing,String}=nothing, funsql_path::Union{Nothing,String}=nothing)
    # If caller provided paths, prefer those
    if synthea_path !== nothing && funsql_path !== nothing
        println("Using provided dataset paths.")
        return synthea_path, funsql_path
    end

    println("Downloading datasets from HuggingFace (or using local cache)...")
    synthea_ds = HF.info(HF.Dataset, "JuliaHealthOrg/JuliaHealthDatasets")
    funsql_ds = HF.info(HF.Dataset, "JuliaHealthOrg/FunSQLQueries")

    synthea_dataset_path = HF.file_download(synthea_ds, "synthea_1M_3YR.duckdb")
    funsql_dataset_path = HF.file_download(funsql_ds, "train.jsonl")

    return synthea_dataset_path, funsql_dataset_path
end

function main()
    model_name = length(ARGS) >= 1 ? ARGS[1] : "meta-llama/Llama-2-7b-chat-hf"
    model_embedding = length(ARGS) >= 2 ? ARGS[2] : "sentence-transformers/all-MiniLM-L6-v2"
    sample_limit = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10

    println("Model: $model_name")
    println("Embedding model: $model_embedding")
    println("Sample limit: $sample_limit")

    # Allow overriding dataset paths via ARGS (positions 4 and 5)
    provided_synthea = length(ARGS) >= 4 ? ARGS[4] : nothing
    provided_funsql = length(ARGS) >= 5 ? ARGS[5] : nothing

    synthea_path, funsql_path = download_datasets(synthea_path=provided_synthea, funsql_path=provided_funsql)

    println("Registering models with PromptingTools...")
    HealthLLM.register_models(model_name, model_embedding)

    # detect number of examples in FunSQL dataset to set progress total
    detected_len = 0
    open(funsql_path, "r") do io
        for line in eachline(io)
            if !isempty(strip(line))
                detected_len += 1
            end
        end
    end
    # if file looks like a single-line JSON array, parse to get length
    if detected_len <= 1
        arr = JSON3.read(read(funsql_path, String))
        try
            detected_len = length(arr)
        catch
            # leave detected_len as-is
        end
    end

    total_examples = sample_limit > 0 ? min(sample_limit, detected_len) : detected_len

    # create ProgressMeter with exact total and register a callback to update it
    pm = Progress(total_examples; show_eta=true)
    HealthLLM.register_progress!((current,total,msg)->begin
        # set the meter to current (clamp to total)
        v = clamp(current, 0, total_examples)
        ProgressMeter.update!(pm, v)
    end)

    println("Starting benchmark run...")
    res = HealthLLM.run_benchmark(model_name, model_embedding, synthea_path, funsql_path; sample_limit=sample_limit)

    println() # newline after progress
    println("Metrics:")
    println(JSON3.write(res["metrics"]))

    failures = res["failures"]
    println("Failures: $(length(failures))")

    out_file = "benchmark_result_$(replace(string(now()), ':' => '-')).json"
    open(out_file, "w") do io
        JSON3.write(io, res)
    end

    println("Wrote full results to: $out_file")
end

main()
