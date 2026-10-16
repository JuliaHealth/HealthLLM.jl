# Check SQL/FunSQL pairs against an OMOP DuckDB database.
#
#   julia --project=scripts/benchmark scripts/benchmark/verify_pairs.jl \
#       pairs.jsonl synthea_1M_3YR.duckdb [report.jsonl]
#
# Each record needs `sql_query` and `response` (FunSQL). For each one this runs
# the SQL, renders the FunSQL against the reflected schema, runs that too, and
# compares the two results with `results_match`. One line is printed per record,
# and with a third argument a JSONL report is written. A pair passes only when
# both queries run and return the same non-empty result. An empty result passes
# with status `empty`, because it can't distinguish a right query from a wrong one.
#
# This is the check used to repair the FunSQL example dataset: write a DuckDB
# port of the SQL and a FunSQL translation, then iterate until every pair passes.

using HealthLLM
using FunSQL
using JSON3

include(joinpath(@__DIR__, "omop_duckdb.jl"))

function verify(path, dbpath; report=nothing, timeout=60)
    (; conn, catalog) = open_omop_duckdb(dbpath)
    run = duckdb_executor(conn; timeout=timeout)
    records = [JSON3.read(l) for l in eachline(path) if !isempty(strip(l))]
    out = report === nothing ? nothing : open(report, "w")
    counts = Dict{String,Int}()
    for (i, rec) in enumerate(records)
        label = string(get(rec, :id, get(rec, :idx, i)))
        sql = String(rec.sql_query)
        gold = try
            run(sql)
        catch err
            err
        end
        status, detail = if !(gold isa ResultSet)
            "sql_error", first(sprint(showerror, gold), 300)
        else
            check = sanity_check_funsql(String(rec.response); catalog=catalog)
            if !check.ok
                "funsql_$(check.stage === :none ? "parse" : check.stage === :parsed ? "build" : "render")_error",
                something(check.error, "")
            else
                pred = try
                    run(check.sql)
                catch err
                    err
                end
                if !(pred isa ResultSet)
                    "funsql_execute_error", first(sprint(showerror, pred), 300)
                elseif !results_match(gold, pred)
                    "mismatch", "sql $(size(gold)) vs funsql $(size(pred)); sql first row: " *
                                repr(isempty(gold.rows) ? nothing : first(gold.rows)) * "; funsql first row: " *
                                repr(isempty(pred.rows) ? nothing : first(pred.rows))
                else
                    isempty(gold.rows) ? "empty" : "ok", "$(size(gold))"
                end
            end
        end
        counts[status] = get(counts, status, 0) + 1
        println(rpad(label, 34), rpad(status, 22), first(replace(detail, r"\s+" => " "), 200))
        flush(stdout)
        out === nothing || (JSON3.write(out, (; id=label, status, detail)); println(out); flush(out))
    end
    out === nothing || close(out)
    println("\n", join(("$k=$v" for (k, v) in sort(collect(counts))), "  "))
end

length(ARGS) in (2, 3) || error("usage: verify_pairs.jl <pairs.jsonl> <omop.duckdb> [report.jsonl]")
verify(ARGS[1], ARGS[2]; report=length(ARGS) == 3 ? ARGS[3] : nothing)
