import HuggingFaceHub as HF
using JSON3
using DuckDB
using JSON3
using Test
using DataFrames
using FunSQL

# This test uses large external datasets and downloads >1GB by default.
# To avoid accidental long-running downloads and excessive disk usage,
# run it explicitly by setting environment variable RUN_HEAVY_TESTS=1.
if get(ENV, "RUN_HEAVY_TESTS", "0") != "1"
    @info "Skipping FunSQLTest (large dataset). Set RUN_HEAVY_TESTS=1 to enable."
else
    # Optionally run a small HuggingFace Inference API check before heavy downloads.
    if get(ENV, "RUN_MODEL_INFERENCE", "0") == "1"
        hf_token = get(ENV, "HF_API_TOKEN", nothing)
        if hf_token === nothing || hf_token == ""
            @warn "RUN_MODEL_INFERENCE requested but HF_API_TOKEN is not set. Skipping model inference step."
        else
            model_name = get(ENV, "HF_MODEL", "gpt2")
            prompt = "Write a short sentence about testing models."

            url = "https://api-inference.huggingface.co/models/" * model_name
            headers = ["Authorization" => "Bearer $hf_token", "Content-Type" => "application/json"]
            body = JSON3.write(Dict("inputs" => prompt))

            try
                # Prefer using HuggingFaceHub.jl helper functions when available.
                parsed = nothing
                made_request = false

                if isdefined(HF, :inference)
                    try
                        parsed = HF.inference(model_name, prompt; token=hf_token)
                        made_request = true
                    catch _
                    end
                end

                if !made_request && isdefined(HF, :text_generation)
                    try
                        parsed = HF.text_generation(model_name; inputs=prompt, token=hf_token)
                        made_request = true
                    catch _
                    end
                end

                # If no convenient wrapper found or wrappers failed, fall back to HTTP API
                if !made_request
                    try
                        using HTTP
                        resp = HTTP.request("POST", url, headers; body=body, retry_limit=2)
                        if resp.status == 200
                            parsed = JSON3.read(String(resp.body))
                            made_request = true
                        else
                            @warn "Model inference request failed (status=$(resp.status)). Response: $(String(resp.body))"
                        end
                    catch err_http
                        @warn "HTTP fallback not available or failed: $err_http"
                    end
                end

                if made_request
                    @test !isempty(string(parsed))
                    @info "HuggingFace inference succeeded for model=$model_name"
                else
                    @warn "Model inference did not produce a response"
                end
            catch err
                @warn "Model inference step failed: $err"
            end
        end
    end
    # Download dataset files
    synthea_ds = HF.info(HF.Dataset, "JuliaHealthOrg/JuliaHealthDatasets")
    funsql_ds = HF.info(HF.Dataset, "JuliaHealthOrg/FunSQLQueries")

    synthea_dataset_path = HF.file_download(synthea_ds, "synthea_1M_3YR.duckdb")
    funsql_dataset_path = HF.file_download(funsql_ds, "train.jsonl")

    # Connect to DuckDB
    conn = DuckDB.DB(synthea_dataset_path)

    # Load FunSQLQueries dataset
    data = JSON3.read(read(funsql_dataset_path, String))

    @testset "FunSQL vs SQL Query Results" begin
        for (i, row) in enumerate(data)
            sql_query = row["sql_query"]
            funsql_code = row["response"]

            try
                # Evaluate FunSQL code to get the query object
                funsql_query = eval(Meta.parse(funsql_code))

                # Render FunSQL query to SQL for DuckDB
                funsql_sql = FunSQL.render(funsql_query, dialect=FunSQL.SQLDialect(:duckdb))

                # Execute both queries
                sql_result = DuckDB.execute(conn, sql_query) |> DataFrame
                funsql_result = DuckDB.execute(conn, funsql_sql) |> DataFrame

                # Compare results
                is_equal = isequal(sql_result, funsql_result)
            catch err
                @warn "Query $i failed: $err"
                is_equal = false
            end

            @testset "Query $i" begin
                @test is_equal
            end
        end
    end
end
