module Benchmark

using JSON3
using DuckDB
using DataFrames
using FunSQL
using Statistics
using Dates
using Tables
using HealthLLM
# import the progress reporter from the sibling Utils module (parent is HealthLLM)
using ..Utils: report_progress

"""
Extract a code block from a model response. If a triple-backtick block exists, return its contents,
otherwise return the whole response trimmed.
"""
function extract_code(resp::AbstractString)
    s = String(resp)
    # Try to find fenced code block
    m = match(r"```(?:julia)?\s*(.*?)\s*```"s, s)
    if m !== nothing
        return strip(m.captures[1])
    end

    # Try to find ``` without closing (fallback)
    m2 = match(r"```(.*)"s, s)
    if m2 !== nothing
        return strip(m2.captures[1])
    end

    return strip(s)
end

function load_jsonl(path::String)
    objs = Vector{Any}()
    open(path, "r") do io
        for line in eachline(io)
            line = strip(line)
            if isempty(line)
                continue
            end
            push!(objs, JSON3.read(line))
        end
    end
    return objs
end

function df_is_numeric_column(colvec)
    # treat as numeric if element type is a subtype of Real or all non-missing elements are Real
    et = eltype(colvec)
    if et <: Real
        return true
    end
    # fallback: inspect values
    for v in colvec
        if v === missing
            continue
        end
        if !(v isa Real)
            return false
        end
    end
    return true
end

function df_equal(df1::DataFrame, df2::DataFrame; atol::Float64=1e-9, rtol::Float64=1e-6, sort_rows::Bool=true)
    # ensure same columns
    cols1 = names(df1)
    cols2 = names(df2)
    if length(cols1) != length(cols2) || Set(cols1) != Set(cols2)
        return false
    end

    # reorder df2 to match df1
    if cols1 != cols2
        df2 = df2[:, cols1]
    end

    # optional: sort rows by all columns to make order-invariant comparisons
    if sort_rows
        # convert to vector of column names for sort
        df1 = sort(df1, cols1)
        df2 = sort(df2, cols1)
    end

    # compare number of rows
    if nrow(df1) != nrow(df2)
        return false
    end

    n = nrow(df1)
    for col in cols1
        col1 = df1[!, col]
        col2 = df2[!, col]

        if df_is_numeric_column(col1) && df_is_numeric_column(col2)
            for i in 1:n
                v1 = col1[i]
                v2 = col2[i]
                if v1 === missing && v2 === missing
                    continue
                elseif v1 === missing || v2 === missing
                    return false
                else
                    # compare numerically
                    if !isapprox(float(v1), float(v2); atol=atol, rtol=rtol)
                        return false
                    end
                end
            end
        else
            # non-numeric: compare with == (handles strings, bools, etc.)
            for i in 1:n
                v1 = col1[i]
                v2 = col2[i]
                if v1 === missing && v2 === missing
                    continue
                elseif v1 === missing || v2 === missing
                    return false
                else
                    if v1 != v2
                        return false
                    end
                end
            end
        end
    end

    return true
end

function choose_nl_field(row::Dict)
    # Candidate keys commonly used for natural-language queries
    candidates = ["question", "query", "nl", "natural_language", "utterance", "text", "english",
                  "question_text", "instruction", "prompt", "input", "content", "message", "messages"]
    # Keys that look like gold answers / should be ignored
    exclude = Set(["sql_query", "sql", "gold_sql", "funsql_code"])

    # Try explicit candidate keys first and ensure they contain non-empty text.
    for k in candidates
        if haskey(row, k)
            v = row[k]
            # plain string
            if isa(v, AbstractString)
                s = strip(String(v))
                if !isempty(s)
                    return s
                end
            # nested object (Dict or NamedTuple) - look for text-like fields
            elseif isa(v, Dict) || isa(v, NamedTuple)
                for kk in ["text", "content", "instruction", "prompt", "utterance"]
                    if haskey(v, kk)
                        s = strip(String(v[kk]))
                        if !isempty(s)
                            return s
                        end
                    end
                end
            # array of messages (e.g. chat format)
            elseif isa(v, AbstractVector) && !isempty(v)
                # collect content/text from elements that are dict-like
                parts = String[]
                for el in v
                    if isa(el, Dict) || isa(el, NamedTuple)
                        if haskey(el, "content")
                            push!(parts, String(el["content"]))
                        elseif haskey(el, "text")
                            push!(parts, String(el["text"]))
                        end
                    elseif isa(el, AbstractString)
                        push!(parts, String(el))
                    end
                end
                if !isempty(parts)
                    s = strip(join(parts, "\n"))
                    if !isempty(s)
                        return s
                    end
                end
            end
        end
    end

    # Fallback: find the first string-like field that doesn't look like gold SQL
    for (k, v) in row
        if k in exclude
            continue
        end
        if isa(v, AbstractString)
            s = strip(String(v))
            if !isempty(s)
                return s
            end
        end
    end

    return nothing
end

"""
Run a benchmark over a FunSQL dataset.

Arguments:
- model_name: name of generator model (HuggingFace id)
- model_embedding: embedding model id for retriever
- synthea_db_path: path to the DuckDB database used to run gold SQL
- funsql_jsonl_path: path to the FunSQL queries dataset (jsonl)

Keyword arguments:
- prompt_template: template string containing `{input_query}` to substitute
- sample_limit: if >0, limit number of evaluated examples

Returns a Dict with metrics and a vector of failure details.
"""
function run_benchmark(model_name::String, model_embedding::String, synthea_db_path::String, funsql_jsonl_path::String; prompt_template::String="Write a valid FunSQL.jl expression that implements the following natural language query:\n\n{input_query}\n\nReturn only the FunSQL.jl expression (no surrounding text).", sample_limit::Int=0)
    # lazy load dataset
    # load as jsonl; if file is a JSON array, fallback to parse whole file
    data = load_jsonl(funsql_jsonl_path)
    if isempty(data)
        # maybe a JSON array
        data = JSON3.read(read(funsql_jsonl_path, String))
    end

    # ensure models are registered with PromptingTools
    HealthLLM.register_models(model_name, model_embedding)

    # Open the DuckDB file in read-only mode to avoid exclusive locks
    # (helps when another process like Git has the file open)
    conn = DuckDB.DB(synthea_db_path; readonly=true)

    total = 0
    correct = 0
    failures = Vector{Dict}()
    example_details = Vector{Dict}()

    start_time = now()

    for (i, row_any) in enumerate(data)
        if sample_limit > 0 && total >= sample_limit
            break
        end

        # JSON3 may return objects as Dict or NamedTuple; normalize
        row = isa(row_any, Dict) ? row_any : Dict(row_any)

        # find NL query
        nl = choose_nl_field(row)
        if nl === nothing
            # try common fallback keys
            if haskey(row, "nl")
                nl = row["nl"]
            elseif haskey(row, "question")
                nl = row["question"]
            elseif haskey(row, "sql_query")
                nl = row["sql_query"]
            else
                @warn "Skipping example $i: no natural language field found"
                continue
            end
        end

        gold_sql = get(row, "sql_query", nothing)
        total += 1

        t0 = now()
        # prepare placeholder for generated code so failures can include it
        funsql_code = ""
        # report progress via the registered progress callback on the HealthLLM.Utils module
        HealthLLM.Utils.report_progress(total+1, length(data); msg = "Running example")
        # Run the example and record timing and status; capture per-example failures but continue
        ok = false
        errstr = nothing
        row_time = 0.0
        try
            # call generator via the HealthLLM package API (uses Query.generate_funsql_query internally)
            resp = HealthLLM.generate_funsql_query(i, model_embedding, model_name, prompt_template, String(nl))
            funsql_code = extract_code(String(resp))

            # Evaluate FunSQL expression to obtain query object
            funsql_query = eval(Meta.parse(funsql_code))

            funsql_sql = FunSQL.render(funsql_query, dialect=FunSQL.SQLDialect(:duckdb))

            # Execute both queries
            if gold_sql === nothing
                @warn "Example $i has no gold SQL; skipping comparison"
                throw(ErrorException("no_gold_sql"))
            end

            sql_result = DuckDB.execute(conn, String(gold_sql)) |> DataFrame
            funsql_result = DuckDB.execute(conn, funsql_sql) |> DataFrame

            # use tolerant comparator
            is_equal = df_equal(sql_result, funsql_result)
            ok = is_equal
        catch err
            ok = false
            errstr = string(err)
        finally
            row_time = Dates.value(now() - t0)/1e9
            push!(example_details, Dict("index"=>i, "duration_seconds"=>row_time, "ok"=>ok, "error"=>errstr, "nl"=>String(nl), "funsql_code"=>funsql_code, "gold_sql"=>gold_sql))
        end

        is_equal = ok

        if is_equal
            correct += 1
        else
            # If not already added (error paths already pushed), add failure details
            if !any(f->f["index"]==i, failures)
                push!(failures, Dict("index"=>i, "nl"=>String(nl), "funsql_code"=>funsql_code, "gold_sql"=>gold_sql))
            end
        end
    end

    elapsed = now() - start_time

    metrics = Dict(
        "total" => total,
        "correct" => correct,
        "accuracy" => total == 0 ? 0.0 : correct/total,
        "elapsed_seconds" => Dates.value(elapsed)/1e9
    )

    return Dict("metrics"=>metrics, "failures"=>failures, "examples"=>example_details)
end

end # module
