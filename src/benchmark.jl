module Benchmark

using JSON3
using DataFrames
using FunSQL
using Statistics
using DuckDB
using RAGTools
using PromptingTools
using ..Query
using ..Grounding
using ..Ingestion

function load_jsonl(path::String)
    objs = Vector{Any}()
    open(path, "r") do io
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            push!(objs, JSON3.read(line))
        end
    end
    if isempty(objs)
        objs = JSON3.read(read(path, String))
    end
    objs
end

function extract_code(resp::AbstractString)
    s = String(resp)
    m = match(r"```(?:julia)?\s*(.*?)\s*```"s, s)
    m !== nothing && return strip(m.captures[1])
    m2 = match(r"```(.*)"s, s)
    m2 !== nothing && return strip(m2.captures[1])
    strip(s)
end

function _is_numeric(col)
    eltype(col) <: Real && return true
    for v in col
        v === missing && continue
        v isa Real && return true
        return false
    end
    return false
end

function df_equal(df1::DataFrame, df2::DataFrame; atol=1e-9, rtol=1e-6, sort_rows=true)
    cols1, cols2 = names(df1), names(df2)
    (length(cols1) != length(cols2) || Set(cols1) != Set(cols2)) && return false
    cols1 != cols2 && (df2 = df2[:, cols1])
    sort_rows && (df1 = sort(df1, cols1); df2 = sort(df2, cols1))
    nrow(df1) != nrow(df2) && return false
    n = nrow(df1)
    for col in cols1
        c1, c2 = df1[!, col], df2[!, col]
        if _is_numeric(c1) && _is_numeric(c2)
            for i in 1:n
                v1, v2 = c1[i], c2[i]
                v1 === missing && v2 === missing && continue
                (v1 === missing || v2 === missing) && return false
                isapprox(float(v1), float(v2); atol, rtol) || return false
            end
        else
            for i in 1:n
                v1, v2 = c1[i], c2[i]
                v1 === missing && v2 === missing && continue
                (v1 === missing || v2 === missing) && return false
                v1 != v2 && return false
            end
        end
    end
    return true
end

function run_zeroshot(model_name::String,
                      funsql_jsonl_path::String;
                      synthea_db_path::Union{String,Nothing}=nothing,
                      sample_limit::Int=0,
                      template::Symbol=:FunSQLQueryDirect)
    Grounding.register_funsql_template_no_context!(name=template)
    gen_schema = Utils.get_schema(nothing, model_name)
    generator_kwargs = (model=model_name, schema=gen_schema, template=template)

    data = load_jsonl(funsql_jsonl_path)
    conn = synthea_db_path !== nothing ? DuckDB.DB(synthea_db_path) : nothing

    total = 0; correct = 0
    failures = Dict[]
    examples = Dict[]
    start_time = time_ns()

    for (i, row_any) in enumerate(data)
        sample_limit > 0 && total >= sample_limit && break
        row = row_any isa Dict ? row_any : Dict(row_any)
        nl = get(row, "query", nothing)
        nl === nothing && continue
        gold_sql = get(row, "sql_query", nothing)
        gold_funsql = get(row, "response", nothing)
        total += 1

        t0 = time_ns()
        funsql_code = ""; ok = false; errstr = nothing
        try
            msg = PromptingTools.aigenerate(String(nl); generator_kwargs...)
            funsql_code = extract_code(msg.content)
            funsql_query = eval(Meta.parse(funsql_code))
            funsql_sql = FunSQL.render(funsql_query, dialect=FunSQL.SQLDialect(:duckdb))
            if conn !== nothing && gold_sql !== nothing
                sql_result = DuckDB.execute(conn, String(gold_sql)) |> DataFrame
                funsql_result = DuckDB.execute(conn, funsql_sql) |> DataFrame
                ok = df_equal(sql_result, funsql_result)
            else
                ok = true
            end
        catch err
            errstr = string(err)
        end
        row_time = (time_ns() - t0) / 1e9
        ok && (correct += 1)
        push!(examples, Dict("index"=>i, "nl"=>nl, "funsql_code"=>funsql_code, "gold_sql"=>gold_sql, "gold_funsql"=>gold_funsql, "ok"=>ok, "error"=>errstr, "duration_seconds"=>row_time))
        if !ok
            push!(failures, Dict("index"=>i, "nl"=>nl, "funsql_code"=>funsql_code, "gold_sql"=>gold_sql, "error"=>errstr))
        end
    end

    elapsed = time_ns() - start_time
    metrics = Dict("total"=>total, "correct"=>correct, "accuracy"=>(total==0 ? 0.0 : correct/total), "elapsed_seconds"=>elapsed/1e9)
    Dict("metrics"=>metrics, "failures"=>failures, "examples"=>examples)
end

function run_rag(model_name::String,
                 model_embedding::String,
                 funsql_jsonl_path::String;
                 synthea_db_path::Union{String,Nothing}=nothing,
                 sample_limit::Int=0,
                 template::Symbol=:FunSQLQueryGeneration)
    Grounding.register_funsql_template!(name=template)
    index = Ingestion.build_grounding_index(; embedder_model=model_embedding, verbose=false)

    data = load_jsonl(funsql_jsonl_path)
    conn = synthea_db_path !== nothing ? DuckDB.DB(synthea_db_path) : nothing

    total = 0; correct = 0
    failures = Dict[]
    examples = Dict[]
    start_time = time_ns()

    for (i, row_any) in enumerate(data)
        sample_limit > 0 && total >= sample_limit && break
        row = row_any isa Dict ? row_any : Dict(row_any)
        nl = get(row, "query", nothing)
        nl === nothing && continue
        gold_sql = get(row, "sql_query", nothing)
        gold_funsql = get(row, "response", nothing)
        total += 1

        t0 = time_ns()
        funsql_code = ""; ok = false; errstr = nothing
        try
            result = Query.generate_funsql_query(index, model_embedding, model_name, String(nl); template=template)
            funsql_code = extract_code(string(result))
            funsql_query = eval(Meta.parse(funsql_code))
            funsql_sql = FunSQL.render(funsql_query, dialect=FunSQL.SQLDialect(:duckdb))
            if conn !== nothing && gold_sql !== nothing
                sql_result = DuckDB.execute(conn, String(gold_sql)) |> DataFrame
                funsql_result = DuckDB.execute(conn, funsql_sql) |> DataFrame
                ok = df_equal(sql_result, funsql_result)
            else
                ok = true
            end
        catch err
            errstr = string(err)
        end
        row_time = (time_ns() - t0) / 1e9
        ok && (correct += 1)
        push!(examples, Dict("index"=>i, "nl"=>nl, "funsql_code"=>funsql_code, "gold_sql"=>gold_sql, "gold_funsql"=>gold_funsql, "ok"=>ok, "error"=>errstr, "duration_seconds"=>row_time))
        if !ok
            push!(failures, Dict("index"=>i, "nl"=>nl, "funsql_code"=>funsql_code, "gold_sql"=>gold_sql, "error"=>errstr))
        end
    end

    elapsed = time_ns() - start_time
    metrics = Dict("total"=>total, "correct"=>correct, "accuracy"=>(total==0 ? 0.0 : correct/total), "elapsed_seconds"=>elapsed/1e9)
    Dict("metrics"=>metrics, "failures"=>failures, "examples"=>examples)
end

end
