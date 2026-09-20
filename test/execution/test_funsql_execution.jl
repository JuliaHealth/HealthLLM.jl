using Test
using DataFrames
using DuckDB
using FunSQL

if !@isdefined(TestHarness)
    include(normpath(joinpath(@__DIR__, "..", "harness", "TestHarness.jl")))
    using .TestHarness
end

@testset "FunSQL & SQL Execution-based Validation" begin
    db = create_test_duckdb()
    ev = ExecutionValidator(db)

    @testset "Simple Table Query (FunSQL vs SQL)" begin
        # SQL version
        sql_query = "SELECT patient_id, first_name, last_name, age FROM patients WHERE age >= 65 ORDER BY patient_id;"
        expected_df = DataFrame(DuckDB.execute(db, sql_query))

        # FunSQL node version with SQLTable
        patients_table = FunSQL.SQLTable(:patients, columns=[:patient_id, :first_name, :last_name, :gender, :birth_date, :age])
        funsql_query = FunSQL.From(patients_table) |>
                       FunSQL.Where(FunSQL.Fun.">="(FunSQL.Get.age, 65)) |>
                       FunSQL.Order(FunSQL.Get.patient_id) |>
                       FunSQL.Select(FunSQL.Get.patient_id, FunSQL.Get.first_name, FunSQL.Get.last_name, FunSQL.Get.age)

        # Validate FunSQL node directly with ExecutionValidator
        res = validate(ev, funsql_query, expected_df)
        @test res.passed
        @test isequal(res.actual, expected_df)
    end

    @testset "Relational Join Query (Patients + Conditions)" begin
        sql_join = """
            SELECT p.patient_id, p.first_name, c.code, c.description
            FROM patients p
            JOIN conditions c ON p.patient_id = c.patient_id
            WHERE c.code = 'I10'
            ORDER BY p.patient_id;
        """
        expected_join_df = DataFrame(DuckDB.execute(db, sql_join))

        # Validate with ExecutionValidator
        res = validate(ev, sql_join, expected_join_df)
        @test res.passed
        @test nrow(expected_join_df) == 2
        @test sort(expected_join_df.patient_id) == [1, 3]
    end

    @testset "Aggregation & Measurement Query" begin
        sql_agg = """
            SELECT p.gender, AVG(o.value_numeric) AS avg_sbp
            FROM patients p
            JOIN observations o ON p.patient_id = o.patient_id
            WHERE o.code = '8480-6'
            GROUP BY p.gender
            ORDER BY p.gender;
        """
        res = validate(ev, sql_agg, df -> nrow(df) == 2 && hasproperty(df, :avg_sbp))
        @test res.passed
    end

    @testset "QueryExecutor directly" begin
        qe = QueryExecutor(db)
        df = execute(qe, "SELECT COUNT(*) AS total FROM patients;")
        @test df.total[1] == 5
    end
end
