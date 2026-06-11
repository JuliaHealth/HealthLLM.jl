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
using HTTP
const HAS_PROGRESS = true

function download_datasets(; synthea_path::Union{Nothing,String}=nothing, funsql_path::Union{Nothing,String}=nothing)
    # Allow using local copies of datasets if present on disk
    local_default_synthea = "E:\\HealthLLM.jl\\synthea_1M_3YR.duckdb"
    if synthea_path === nothing && isfile(local_default_synthea)
        println("Found local Synthea duckdb at: $local_default_synthea; using it.")
        synthea_path = local_default_synthea
    end

    # Prefer a locally checked-out FunSQL dataset when available
    local_default_funsql = "E:\\HealthLLM.jl\\FunSQLQueries\\train.jsonl"
    if funsql_path === nothing && isfile(local_default_funsql)
        println("Found local FunSQL dataset at: $local_default_funsql; using it.")
        funsql_path = local_default_funsql
    end

    # If caller provided both paths, prefer those
    if synthea_path !== nothing && funsql_path !== nothing
        println("Using provided dataset paths.")
        return synthea_path, funsql_path
    end

    println("Downloading datasets from HuggingFace (or using local cache)...")
    synthea_name = "JuliaHealthOrg/JuliaHealthDatasets"
    funsql_name = "JuliaHealthOrg/FunSQLQueries"
    synthea_ds = HF.info(HF.Dataset, synthea_name)
    funsql_ds = HF.info(HF.Dataset, funsql_name)

    # show which files will be downloaded and where they will be stored
    println("Will download (only missing files):")
    println(" - $synthea_name -> synthea_1M_3YR.duckdb")
    println(" - $funsql_name -> train.jsonl")

    # helper: attempt a streaming HTTP download with per-file progress
    function try_stream_download(repo::String, filename::String)
        candidates = [
            "https://huggingface.co/datasets/$repo/resolve/main/$filename",
            "https://huggingface.co/$repo/resolve/main/$filename",
            "https://huggingface.co/datasets/$repo/resolve/refs/heads/main/$filename"
        ]

        for url in candidates
            try
                # Try HEAD first to learn content length
                head_res = try
                    HTTP.request("HEAD", url)
                catch
                    nothing
                end

                total_bytes = nothing
                if head_res !== nothing && head_res.status == 200
                    clen = get(head_res.headers, "Content-Length", nothing)
                    if clen !== nothing
                        try
                            total_bytes = parse(Int, String(clen))
                        catch
                            total_bytes = nothing
                        end
                    end
                end

                dest = abspath(filename)
                println("Attempting download: $filename from $url -> $dest")

                # Create a Progress instance in a way compatible with multiple
                # ProgressMeter versions. Older versions may not accept the
                # `show_eta` keyword, so try the full call first and fall back
                # to a minimal constructor when necessary.
                function _make_progress(n)
                    try
                        return Progress(n; show_eta=true)
                    catch
                        return Progress(n)
                    end
                end

                pm_file = total_bytes !== nothing ? _make_progress(total_bytes) : _make_progress(1)

                # Stream GET
                try
                    HTTP.open(:GET, url) do stream_io
                        open(dest, "w") do out_io
                            bytes_written = 0
                            while !eof(stream_io)
                                chunk = read(stream_io, 65536)
                                if isempty(chunk)
                                    break
                                end
                                write(out_io, chunk)
                                bytes_written += length(chunk)
                                try
                                    if total_bytes !== nothing
                                        ProgressMeter.update!(pm_file, bytes_written)
                                    else
                                        ProgressMeter.update!(pm_file)
                                    end
                                catch
                                    # ignore progress update failures
                                end
                            end
                        end
                    end
                catch err
                    @warn "GET streaming failed for $url: $err"
                    continue
                end

                println("Finished downloading $filename to $dest")
                return dest
            catch err
                @warn "Stream download attempt failed for $url: $err"
            end
        end

        # fallback to HuggingFaceHub.file_download if available
        try
            if isdefined(HF, :file_download)
                info = HF.info(HF.Dataset, repo)
                dest = HF.file_download(info, filename)
                println("Downloaded $filename via HuggingFaceHub.file_download -> $dest")
                return dest
            end
        catch err
            @warn "HuggingFaceHub.file_download fallback failed: $err"
        end

        error("Could not download $filename from repository $repo")
    end

    # overall download progress across both dataset files
    # Use the same compatibility helper to construct the meter.
    function _make_progress(n)
        try
            return Progress(n; show_eta=true)
        catch
            return Progress(n)
        end
    end

    # Determine how many files we actually need to download
    need_synthea = synthea_path === nothing
    need_funsql = funsql_path === nothing
    total_to_download = (need_synthea ? 1 : 0) + (need_funsql ? 1 : 0)
    pm = _make_progress(max(total_to_download, 1))

    synthea_dataset_path = synthea_path
    if need_synthea
        synthea_dataset_path = try_stream_download(synthea_name, "synthea_1M_3YR.duckdb")
        ProgressMeter.update!(pm, 1)
    end

    funsql_dataset_path = funsql_path
    if need_funsql
        # if we already downloaded synthea, update progress position for the second file
        offset = need_synthea ? 2 : 1
        funsql_dataset_path = try_stream_download(funsql_name, "train.jsonl")
        ProgressMeter.update!(pm, offset)
    end

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
        detected_len = length(arr)
    end

    total_examples = sample_limit > 0 ? min(sample_limit, detected_len) : detected_len

    # create ProgressMeter with exact total and register a callback to update it
    # Use a compatibility constructor to avoid passing unsupported keywords.
    function _make_progress(n)
        try
            return Progress(n; show_eta=true)
        catch
            return Progress(n)
        end
    end

    pm = _make_progress(total_examples)
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
