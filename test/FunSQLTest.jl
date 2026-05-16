import HuggingFaceHub as HF
using DuckDB
using JSON3
using Test
using DataFrames
using FunSQL

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

if !haskey(ENV, "SSL_CERT_FILE")
    cacert_path = joinpath(dirname(@__DIR__), "cacert.pem")
    if isfile(cacert_path)
        ENV["SSL_CERT_FILE"] = cacert_path
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

try
    @testset "FunSQL vs SQL Query Results" begin
        for (i, row) in enumerate(data)
            sql_query = row["sql_query"]
            funsql_code = row["response"]

            try
                funsql_expr = Meta.parse(funsql_code)
                assert_safe_funsql_expr(funsql_expr)

                # Evaluate the expression inside a dedicated sandbox module.
                funsql_query = Core.eval(FUNSQL_EVAL_MODULE, funsql_expr)

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
finally
    DuckDB.close(conn)
end
