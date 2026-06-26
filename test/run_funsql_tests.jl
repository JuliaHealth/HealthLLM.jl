using DuckDB
using JSON3
using DataFrames
using FunSQL
using Test
using HuggingFaceHub
const HF = HuggingFaceHub
using Printf
using MozillaCACerts_jll

for var in ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]
    if !haskey(ENV, var)
        ENV[var] = var in ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"] ? 
                   MozillaCACerts_jll.cacert : dirname(MozillaCACerts_jll.cacert)
    end
end

const DISALLOWED_EXPR_HEADS = Set([
    :module, :toplevel, :using, :import, :macrocall, :function, :->, :for, :while,
    :let, :quote, :global, :local, :const, :try, :return, :break, :continue, :ccall
])

function assert_safe_funsql_expr(expr)
    expr isa Expr || return nothing
    expr.head in DISALLOWED_EXPR_HEADS && throw(
        ArgumentError("Disallowed expression head in FunSQL expression: $(expr.head)")
    )
    for arg in expr.args
        assert_safe_funsql_expr(arg)
    end
    return nothing
end

const FUNSQL_EVAL_MODULE = Module(:FunSQLEvalSandbox)
Core.eval(FUNSQL_EVAL_MODULE, :(using FunSQL))

function download_datasets()
    println("Downloading datasets from HuggingFace...")
    
    println("  - Downloading FunSQLQueries dataset...")
    funsql_ds = HF.info(HF.Dataset, "JuliaHealthOrg/FunSQLQueries")
    funsql_path = HF.file_download(funsql_ds, "train.jsonl")
    
    println("  - Downloading Synthea database...")
    synthea_ds = HF.info(HF.Dataset, "JuliaHealthOrg/JuliaHealthDatasets")
    synthea_path = HF.file_download(synthea_ds, "synthea_1M_3YR.duckdb")
    
    return funsql_path, synthea_path
end

function load_duckdb(db_path::String)
    return DuckDB.DB(db_path)
end

function parse_funsql(funsql_code::String)
    funsql_expr = Meta.parse(funsql_code)
    assert_safe_funsql_expr(funsql_expr)
    funsql_query = Core.eval(FUNSQL_EVAL_MODULE, funsql_expr)
    return funsql_query
end

function execute_funsql(conn, funsql_code::String; dialect::Symbol=:duckdb)
    funsql_query = parse_funsql(funsql_code)
    funsql_sql = FunSQL.render(funsql_query, dialect=FunSQL.SQLDialect(dialect))
    result = DuckDB.execute(conn, funsql_sql) |> DataFrame
    return result
end

function execute_sql(conn, sql_query::String)
    result = DuckDB.execute(conn, sql_query) |> DataFrame
    return result
end

function compare_results(df1::DataFrame, df2::DataFrame)
    return isequal(df1, df2)
end

function run_tests(; funsql_path=nothing, db_path=nothing, max_tests::Int=0)
    if funsql_path === nothing || db_path === nothing
        println("\nNote: Provide paths via environment variables to skip download:")
        println("  FUNSQL_DATA_PATH=/path/to/train.jsonl")
        println("  SYNTHEA_DB_PATH=/path/to/synthea_1M_3YR.duckdb")
        println()
        funsql_path, db_path = download_datasets()
    end
    
    if !isfile(funsql_path)
        error("FunSQL data file not found: $funsql_path")
    end
    
    if !isfile(db_path)
        error("Synthea database file not found: $db_path")
    end
    
    println("\nLoading dataset from: $funsql_path")
    data = JSON3.read(read(funsql_path, String))
    
    if max_tests > 0
        data = data[1:min(max_tests, length(data))]
    end
    
    println("Connecting to database: $db_path")
    conn = load_duckdb(db_path)
    
    println("\n" * "="^60)
    println("FunSQL Validation Tests")
    println("="^60)
    
    total_tests = length(data)
    passed = 0
    failed = 0
    errors = []
    
    try
        for (i, row) in enumerate(data)
            sql_query = row["sql_query"]
            funsql_code = row["response"]
            
            println("\n--- Test $i ---")
            println("FunSQL: $(first(strip(funsql_code), 80))...")
            
            try
                funsql_result = execute_funsql(conn, funsql_code)
                sql_result = execute_sql(conn, sql_query)
                
                is_equal = compare_results(sql_result, funsql_result)
                
                if is_equal
                    println("✓ PASSED")
                    passed += 1
                else
                    println("✗ FAILED")
                    println("  SQL rows: $(nrow(sql_result)), FunSQL rows: $(nrow(funsql_result))")
                    push!(errors, (i, "Row count mismatch"))
                    failed += 1
                end
            catch err
                println("✗ ERROR: $err")
                push!(errors, (i, string(err)))
                failed += 1
            end
            
            if i % 10 == 0
                @printf "Progress: %d/%d (%.1f%%)\n" i total_tests (i/total_tests*100)
            end
        end
    finally
        DuckDB.close(conn)
    end
    
    println("\n" * "="^60)
    println("Test Results Summary")
    println("="^60)
    println("Total:  $total_tests")
    println("Passed: $passed")
    println("Failed: $failed")
    println("Success Rate: $(round(passed/total_tests*100, digits=1))%")
    
    if length(errors) > 0
        println("\nFailed Tests:")
        for (test_num, error_msg) in errors[1:min(5, length(errors))]
            println("  Test $test_num: $(error_msg)")
        end
        if length(errors) > 5
            println("  ... and $(length(errors) - 5) more failures")
        end
    end
    
    return passed, failed, total_tests
end

function main(; max_tests::Int=0)
    passed, failed, total = run_tests(max_tests=max_tests)
    
    if failed > 0
        error("$failed out of $total tests failed!")
    else
        println("\n✓ All $total tests passed!")
    end
end

const USAGE = """
Usage: julia --project=. test/run_funsql_tests.jl [max_tests]

Arguments:
  max_tests  - Optional. Maximum number of tests to run (0 = all)

Environment Variables:
  SYNTHEA_DB_PATH    - Path to Synthea DuckDB file
  FUNSQL_DATA_PATH   - Path to FunSQL queries JSONL file
"""

if abspath(PROGRAM_FILE) == @__FILE__
    max_tests = 0
    if length(ARGS) > 0
        max_tests = parse(Int, ARGS[1])
    end
    
    if "--help" in ARGS || "-h" in ARGS
        println(USAGE)
    else
        funsql_path = get(ENV, "FUNSQL_DATA_PATH", nothing)
        db_path = get(ENV, "SYNTHEA_DB_PATH", nothing)
        
        run_tests(funsql_path=funsql_path, db_path=db_path, max_tests=max_tests)
    end
end

if !isinteractive()
    main()
end