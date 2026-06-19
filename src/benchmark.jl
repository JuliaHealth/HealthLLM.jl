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
using ..Utils

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

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

function compute_retrieval_stats(examples::Vector)
    total_chunks = 0
    sources_set = Set{String}()
    scores = Float64[]
    n = length(examples)
    for ex in examples
        srcs = get(ex, "sources", nothing)
        if srcs !== nothing && isa(srcs, AbstractVector)
            total_chunks += length(srcs)
            for s in srcs
                push!(sources_set, string(s))
            end
        end
        sc = get(ex, "candidate_scores", nothing)
        if sc !== nothing && isa(sc, AbstractVector)
            for s in sc
                push!(scores, Float64(s))
            end
        end
    end
    score_stats = if isempty(scores)
        Dict("min"=>0.0, "max"=>0.0, "mean"=>0.0, "std"=>0.0, "count"=>0)
    else
        Dict("min"=>round(minimum(scores), digits=6),
             "max"=>round(maximum(scores), digits=6),
             "mean"=>round(mean(scores), digits=6),
             "std"=>length(scores) > 1 ? round(std(scores), digits=6) : 0.0,
             "count"=>length(scores))
    end
    Dict("total_chunks_retrieved"=>total_chunks,
         "avg_chunks_per_query"=>n > 0 ? round(total_chunks / n, digits=2) : 0.0,
         "unique_sources"=>collect(sources_set),
         "num_unique_sources"=>length(sources_set),
         "score_stats"=>score_stats)
end

function compute_detailed_metrics(examples::Vector)
    total = length(examples)
    total == 0 && return Dict()
    correct = count(e -> get(e, "ok", false) == true, examples)
    parse_ok = count(e -> get(e, "parse_ok", false) == true, examples)
    exec_ok = count(e -> get(e, "exec_ok", false) == true, examples)
    group_acc = Dict{String,Dict{String,Any}}()
    for e in examples
        grp = string(get(e, "group", "unknown"))
        if !haskey(group_acc, grp)
            group_acc[grp] = Dict("total"=>0, "correct"=>0, "parse_ok"=>0)
        end
        group_acc[grp]["total"] += 1
        get(e, "ok", false) == true && (group_acc[grp]["correct"] += 1)
        get(e, "parse_ok", false) == true && (group_acc[grp]["parse_ok"] += 1)
    end
    group_results = Dict{String,Dict}()
    for (g, d) in sort!(collect(group_acc), by=x->x[2]["total"], rev=true)
        t = d["total"]
        group_results[string(g)] = Dict("total"=>t, "correct"=>d["correct"],
            "parse_ok"=>d["parse_ok"],
            "accuracy"=>t > 0 ? round(d["correct"] / t, digits=4) : 0.0)
    end
    error_types = Dict("parse_error"=>0, "execution_error"=>0, "result_mismatch"=>0)
    for e in examples
        get(e, "ok", false) == true && continue
        if get(e, "parse_ok", false) == false
            error_types["parse_error"] += 1
        elseif get(e, "exec_ok", false) == false
            error_types["execution_error"] += 1
        else
            error_types["result_mismatch"] += 1
        end
    end
    total_duration = sum(e -> get(e, "duration_seconds", 0.0), examples)
    Dict("total"=>total, "correct"=>correct,
         "accuracy"=>total > 0 ? round(correct / total, digits=4) : 0.0,
         "parse_success_rate"=>total > 0 ? round(parse_ok / total, digits=4) : 0.0,
         "execution_success_rate"=>total > 0 ? round(exec_ok / total, digits=4) : 0.0,
         "accuracy_by_group"=>group_results,
         "error_type_breakdown"=>error_types,
         "total_duration_seconds"=>round(total_duration, digits=2),
         "avg_duration_per_query"=>total > 0 ? round(total_duration / total, digits=3) : 0.0)
end

function compute_comparison(zero_examples::Vector, rag_examples::Vector)
    z_metrics = compute_detailed_metrics(zero_examples)
    r_metrics = compute_detailed_metrics(rag_examples)
    r_retrieval = compute_retrieval_stats(rag_examples)
    z_acc = get(z_metrics, "accuracy", 0.0)
    r_acc = get(r_metrics, "accuracy", 0.0)
    n = min(length(zero_examples), length(rag_examples))
    both_right = 0; both_wrong = 0; rag_only = 0; zero_only = 0
    per_query = Dict[]
    for i in 1:n
        ze = zero_examples[i]; re = rag_examples[i]
        z_ok = get(ze, "ok", false) == true
        r_ok = get(re, "ok", false) == true
        both_right += (z_ok && r_ok) ? 1 : 0
        both_wrong += (!z_ok && !r_ok) ? 1 : 0
        rag_only += (!z_ok && r_ok) ? 1 : 0
        zero_only += (z_ok && !r_ok) ? 1 : 0
        push!(per_query, Dict("index"=>get(ze, "index", i),
            "nl"=>get(ze, "nl", ""), "zero_ok"=>z_ok, "rag_ok"=>r_ok,
            "zero_error"=>get(ze, "error", nothing),
            "rag_error"=>get(re, "error", nothing)))
    end
    Dict("zero_shot"=>z_metrics, "rag"=>merge(r_metrics, Dict("retrieval"=>r_retrieval)),
         "comparison"=>Dict("accuracy_delta"=>round(r_acc - z_acc, digits=4),
            "both_correct"=>both_right, "both_wrong"=>both_wrong,
            "rag_correct_only"=>rag_only, "zero_shot_correct_only"=>zero_only,
            "total_compared"=>n, "per_query"=>per_query))
end

# ---------------------------------------------------------------------------
# Execution functions
# ---------------------------------------------------------------------------

function _execute_and_compare(conn, gold_sql, funsql_sql)
    sql_result = DuckDB.execute(conn, String(gold_sql)) |> DataFrame
    funsql_result = DuckDB.execute(conn, funsql_sql) |> DataFrame
    return df_equal(sql_result, funsql_result)
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
    examples = Dict[]
    start_time = time_ns()

    for (i, row_any) in enumerate(data)
        sample_limit > 0 && total >= sample_limit && break
        row = row_any isa Dict ? row_any : Dict(row_any)
        nl = get(row, "query", nothing)
        nl === nothing && continue
        gold_sql = get(row, "sql_query", nothing)
        gold_funsql = get(row, "response", nothing)
        grp = get(row, "group", "unknown")
        total += 1

        t0 = time_ns()
        funsql_code = ""; ok = false; parse_ok = false; exec_ok = false
        errstr = nothing; funsql_sql = nothing
        try
            msg = PromptingTools.aigenerate(String(nl); generator_kwargs...)
            funsql_code = extract_code(msg.content)
            parse_ok = true
            funsql_query = eval(Meta.parse(funsql_code))
            funsql_sql = FunSQL.render(funsql_query, dialect=FunSQL.SQLDialect(:duckdb))
            exec_ok = true
            if conn !== nothing && gold_sql !== nothing
                ok = _execute_and_compare(conn, gold_sql, funsql_sql)
            else
                ok = true
            end
        catch err
            errstr = string(err)
        end
        row_time = (time_ns() - t0) / 1e9
        ok && (correct += 1)
        push!(examples, Dict("index"=>i, "group"=>grp, "nl"=>nl,
            "funsql_code"=>funsql_code, "funsql_sql"=>funsql_sql,
            "gold_sql"=>gold_sql, "gold_funsql"=>gold_funsql,
            "ok"=>ok, "parse_ok"=>parse_ok, "exec_ok"=>exec_ok,
            "error"=>errstr, "duration_seconds"=>row_time))
    end

    elapsed = time_ns() - start_time
    detailed = compute_detailed_metrics(examples)
    detailed["elapsed_seconds"] = round(elapsed / 1e9, digits=2)
    Dict("metrics"=>detailed, "examples"=>examples)
end

function run_rag(model_name::String,
                 model_embedding::String,
                 funsql_jsonl_path::String;
                 synthea_db_path::Union{String,Nothing}=nothing,
                 sample_limit::Int=0,
                 template::Symbol=:FunSQLQueryGeneration)
    Grounding.register_funsql_template!(name=template)

    gen_schema = Utils.get_schema(nothing, model_name)
    emb_schema = Utils.get_schema(nothing, model_embedding)
    retriever_kwargs = (model=model_embedding, schema=emb_schema,
        embedder_kwargs=(schema=emb_schema, model=model_embedding))
    generator_kwargs = (model=model_name, schema=gen_schema, template=template)

    index = Ingestion.build_grounding_index(; embedder_model=model_embedding, verbose=false)

    data = load_jsonl(funsql_jsonl_path)
    conn = synthea_db_path !== nothing ? DuckDB.DB(synthea_db_path) : nothing

    total = 0; correct = 0
    examples = Dict[]
    start_time = time_ns()

    for (i, row_any) in enumerate(data)
        sample_limit > 0 && total >= sample_limit && break
        row = row_any isa Dict ? row_any : Dict(row_any)
        nl = get(row, "query", nothing)
        nl === nothing && continue
        gold_sql = get(row, "sql_query", nothing)
        gold_funsql = get(row, "response", nothing)
        grp = get(row, "group", "unknown")
        total += 1

        t0 = time_ns()
        funsql_code = ""; ok = false; parse_ok = false; exec_ok = false
        errstr = nothing; funsql_sql = nothing
        sources = nothing; candidate_scores = nothing
        try
            rag_result = RAGTools.airag(index;
                question=String(nl),
                retriever_kwargs=retriever_kwargs,
                generator_kwargs=generator_kwargs,
                return_all=true)
            answer_text = rag_result.final_answer !== nothing ? rag_result.final_answer :
                          rag_result.answer
            sources = isempty(rag_result.sources) ? nothing : collect(rag_result.sources)
            emb = rag_result.emb_candidates
            candidate_scores = emb !== nothing && !isempty(emb.scores) ?
                               collect(Float64.(emb.scores)) : nothing
            if answer_text !== nothing
                funsql_code = extract_code(string(answer_text))
                parse_ok = true
                funsql_query = eval(Meta.parse(funsql_code))
                funsql_sql = FunSQL.render(funsql_query, dialect=FunSQL.SQLDialect(:duckdb))
                exec_ok = true
                if conn !== nothing && gold_sql !== nothing
                    ok = _execute_and_compare(conn, gold_sql, funsql_sql)
                else
                    ok = true
                end
            end
        catch err
            errstr = string(err)
        end
        row_time = (time_ns() - t0) / 1e9
        ok && (correct += 1)
        push!(examples, Dict("index"=>i, "group"=>grp, "nl"=>nl,
            "funsql_code"=>funsql_code, "funsql_sql"=>funsql_sql,
            "gold_sql"=>gold_sql, "gold_funsql"=>gold_funsql,
            "ok"=>ok, "parse_ok"=>parse_ok, "exec_ok"=>exec_ok,
            "error"=>errstr, "duration_seconds"=>row_time,
            "sources"=>sources, "candidate_scores"=>candidate_scores))
    end

    elapsed = time_ns() - start_time
    detailed = compute_detailed_metrics(examples)
    detailed["elapsed_seconds"] = round(elapsed / 1e9, digits=2)
    detailed["retrieval"] = compute_retrieval_stats(examples)
    Dict("metrics"=>detailed, "examples"=>examples)
end

function run_comparison(model_name::String,
                        model_embedding::String,
                        funsql_jsonl_path::String;
                        synthea_db_path::Union{String,Nothing}=nothing,
                        sample_limit::Int=0)
    z_result = run_zeroshot(model_name, funsql_jsonl_path;
        synthea_db_path=synthea_db_path, sample_limit=sample_limit)
    r_result = run_rag(model_name, model_embedding, funsql_jsonl_path;
        synthea_db_path=synthea_db_path, sample_limit=sample_limit)
    comparison = compute_comparison(
        z_result["examples"], r_result["examples"])
    Dict("comparison"=>comparison, "zeroshot"=>z_result, "rag"=>r_result)
end

end
