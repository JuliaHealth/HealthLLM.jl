# Build the committed benchmark task set from the raw FunSQL example dataset.
#
#   julia --project=scripts/benchmark scripts/benchmark/curate_tasks.jl \
#       FunSQLQueries/train.jsonl synthea_1M_3YR.duckdb data/benchmark/funsql_tasks.jsonl
#
# A task is kept only if its gold SQL runs on the OMOP DuckDB within the timeout
# and returns at least one row. An empty gold result would be "matched" by any
# query that also returns nothing, so it can't tell a right query from a wrong
# one. Each kept task's gold FunSQL is checked too: it must render against the
# reflected schema and return the same result as the gold SQL. When it doesn't,
# the FunSQL is dropped from the task (`response` = ""), so a broken reference
# can never be retrieved as a RAG example.

using HealthLLM
using HealthLLM.Evaluation: _fnv1a
using FunSQL
using JSON3

include(joinpath(@__DIR__, "omop_duckdb.jl"))

function main(src, dbpath, out; timeout=60)
    (; conn, catalog) = open_omop_duckdb(dbpath)
    run = duckdb_executor(conn; timeout=timeout)
    records = [JSON3.read(l) for l in eachline(src) if !isempty(strip(l))]

    kept = BenchmarkTask[]
    status = Dict{String,Int}()
    funsql_ok = 0
    seen = Set{String}()
    for (i, rec) in enumerate(records)
        question = strip(string(get(rec, :query, "")))
        sql = strip(string(get(rec, :sql_query, "")))
        group = strip(string(get(rec, :group, "unknown")))
        (isempty(question) || isempty(sql)) && (status["missing fields"] = get(status, "missing fields", 0) + 1; continue)

        t0 = time()
        gold = try
            run(sql)
        catch err
            err
        end
        why = gold isa QueryTimeout ? "gold SQL timed out" :
              gold isa Exception ? "gold SQL failed on DuckDB" :
              isempty(gold.rows) ? "gold SQL returned no rows" : "kept"
        status[why] = get(status, why, 0) + 1
        println(lpad(i, 3), "  ", rpad(why, 26), round(time() - t0; digits=1), "s  ", first(question, 60))
        why == "kept" || continue

        funsql = strip(string(get(rec, :response, "")))
        verified = false
        if !isempty(funsql)
            check = sanity_check_funsql(funsql; catalog=catalog)
            if check.ok
                pred = try
                    run(check.sql)
                catch
                    nothing
                end
                verified = pred isa ResultSet && results_match(gold, pred)
            end
        end
        verified && (funsql_ok += 1)

        id = string(group, "-", string(_fnv1a(question); base=16)[1:8])
        id in seen && (id *= "-$i")
        push!(seen, id)
        push!(kept, BenchmarkTask(id, question, group, classify_tier(sql), sql,
            verified ? funsql : "", strip(string(get(rec, :description, "")))))
    end

    save_tasks(out, kept)
    println("\n", length(records), " source records -> ", length(kept), " tasks written to ", out)
    for (k, v) in sort(collect(status); by=last, rev=true)
        println("  ", rpad(k, 28), v)
    end
    println("  tiers: ", join(("$t=$(count(x -> x.tier === t, kept))" for t in TIERS), ", "))
    println("  gold FunSQL verified against gold SQL: ", funsql_ok, "/", length(kept))
end

length(ARGS) == 3 || error("usage: curate_tasks.jl <source.jsonl> <omop.duckdb> <out.jsonl>")
main(ARGS...)
