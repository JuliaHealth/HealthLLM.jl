import HuggingFaceHub as HF
using DuckDB
using JSON3
using Test
using DataFrames
using FunSQL


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