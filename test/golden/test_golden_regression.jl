using Test
using DataFrames
using DuckDB
using FunSQL
using HealthLLM
using HealthLLM.Evaluation: BenchmarkTask, run_task, ResultSet, results_match
using HealthLLM.Execution: extract_funsql

if !@isdefined(TestHarness)
    include(normpath(joinpath(@__DIR__, "..", "harness", "TestHarness.jl")))
    using .TestHarness
end

@testset "Golden Reference Regression Suite" begin
    fixtures_dir = joinpath(@__DIR__, "..", "fixtures")

    @testset "End-to-End FunSQL run_task Golden Pipeline" begin
        db = create_test_duckdb()
        # Insert a boundary patient with age 65 to strictly distinguish `>` from `>=`
        DuckDB.execute(db, "INSERT INTO patients VALUES (6, 'Frank', 'Miller', 'M', '1959-01-01', 65);")

        patients_tbl = FunSQL.SQLTable(:patients, columns=[:patient_id, :first_name, :last_name, :gender, :birth_date, :age])
        test_catalog = FunSQL.SQLCatalog(patients_tbl; dialect=:duckdb)

        executor = sql -> begin
            df = DataFrame(DuckDB.execute(db, sql))
            ResultSet(names(df), [collect(Any, r) for r in eachrow(df)])
        end

        # Natural language question: "Find all patients older than 65 years"
        # Reference SQL & FunSQL: strictly `age > 65`
        # Frank (age 65) is excluded; Alice (69) and Charlie (76) are included.
        expected_rs = ResultSet(["patient_id", "first_name"], [[1, "Alice"], [3, "Charlie"]])
        gold_sql = "SELECT patient_id, first_name FROM patients WHERE age > 65 ORDER BY patient_id;"
        gold_funsql = "From(:patients) |> Where(Fun.\">\"(Get.age, 65)) |> Order(Get.patient_id) |> Select(Get.patient_id, Get.first_name)"
        task = BenchmarkTask("golden-funsql-1", "Find all patients older than 65 years", "patients", :simple, gold_sql, gold_funsql, "Filter patients older than 65 years")

        # Injected generator returning valid FunSQL in a julia codeblock
        good_reply = """
        ```julia
        From(:patients) |> Where(Fun.">"(Get.age, 65)) |> Order(Get.patient_id) |> Select(Get.patient_id, Get.first_name)
        ```
        Filters patients strictly older than 65 years.
        """
        injected_gen = (prompt; model, schema=nothing, kwargs...) -> good_reply

        # 1. Real run_task pipeline execution
        result = HealthLLM.Evaluation.run_task(
            task, :zero_shot;
            model="golden-eval-model",
            executor=executor,
            catalog=test_catalog,
            generator=injected_gen
        )

        @test result isa HealthLLM.Evaluation.TaskResult
        @test result.outcome === :correct
        @test result.funsql == "From(:patients) |> Where(Fun.\">\"(Get.age, 65)) |> Order(Get.patient_id) |> Select(Get.patient_id, Get.first_name)"
        @test results_match(expected_rs, executor(result.sql); ordered=true)

        # 2. Operator Precision Regression Protection: `>= 65` mistakenly returned for "older than 65"
        # Frank (age 65) is erroneously included, producing 3 rows instead of 2. Must produce :mismatch!
        wrong_op_gen = (prompt; kwargs...) -> "```julia\nFrom(:patients) |> Where(Fun.\">=\"(Get.age, 65)) |> Order(Get.patient_id) |> Select(Get.patient_id, Get.first_name)\n```"
        r_op = run_task(task, :zero_shot; model="fake", executor=executor, catalog=test_catalog, generator=wrong_op_gen)
        @test r_op.outcome === :mismatch
        @test occursin("result differs from gold", r_op.error)

        # 3. Pipeline Regression Protection: Syntax error in model reply
        bad_syntax_gen = (prompt; kwargs...) -> "```julia\nFrom(:patients) |> Select(\n```"
        r_parse = run_task(task, :zero_shot; model="fake", executor=executor, catalog=test_catalog, generator=bad_syntax_gen)
        @test r_parse.outcome === :parse_error

        # 4. Pipeline Regression Protection: Hallucinated column caught by catalog render
        hallucinated_gen = (prompt; kwargs...) -> "```julia\nFrom(:patients) |> Select(Get.non_existent_column)\n```"
        r_render = run_task(task, :zero_shot; model="fake", executor=executor, catalog=test_catalog, generator=hallucinated_gen)
        @test r_render.outcome === :render_error

        # 5. Pipeline Regression Protection: Semantic mismatch (e.g. inverted filter condition)
        wrong_data_gen = (prompt; kwargs...) -> "```julia\nFrom(:patients) |> Where(Fun.\"<\"(Get.age, 30)) |> Order(Get.patient_id) |> Select(Get.patient_id, Get.first_name)\n```"
        r_mismatch = run_task(task, :zero_shot; model="fake", executor=executor, catalog=test_catalog, generator=wrong_data_gen)
        @test r_mismatch.outcome === :mismatch
    end

    @testset "Dialect Preservation" begin
        patients_tbl = FunSQL.SQLTable(:patients, columns=[:patient_id, :first_name])
        # MySQL dialect renders identifiers with backticks (`patients_1`.`patient_id`),
        # whereas DuckDB dialect renders with double quotes ("patients_1"."patient_id").
        mysql_dialect = FunSQL.SQLDialect(:mysql)
        captured_sql = Ref{String}("")
        mock_executor = sql -> begin
            captured_sql[] = sql
            DataFrame(patient_id=[1])
        end

        ev_mysql = ExecutionValidator(mock_executor; dialect=mysql_dialect)
        @test ev_mysql.dialect === mysql_dialect

        q = FunSQL.From(patients_tbl) |> FunSQL.Select(FunSQL.Get.patient_id)
        res = validate(ev_mysql, q)
        @test res.passed
        # Prove the dialect affected the actual rendered SQL:
        @test occursin("`patients_1`.`patient_id`", captured_sql[])
        @test occursin("FROM `patients`", captured_sql[])
        @test !occursin("\"patients_1\"", captured_sql[])
    end

    @testset "FunSQL Extraction Helper Verification" begin
        # Verify extract_funsql correctly strips various code fence formats supported by the harness
        @test extract_funsql("```julia\nFrom(:p)\n```") == "From(:p)"
        @test extract_funsql("```jl\nFrom(:p)\n```") == "From(:p)"
        @test extract_funsql("```sql\nSELECT 1;\n```") == "SELECT 1;"
        @test extract_funsql("```\nFrom(:p) |> Select(:id)\n```") == "From(:p) |> Select(:id)"
        @test extract_funsql("  From(:p)  ") == "From(:p)"
        @test extract_funsql("Here is the query:\n```julia\nFrom(:p)\n```\nExplanation.") == "From(:p)"
    end

    @testset "Fixture-based Validator Coverage (SQL Queries)" begin
        query_cases_file = joinpath(fixtures_dir, "golden_query_cases.json")
        cases = load_golden_cases(query_cases_file)
        @test !isempty(cases)

        db = create_test_duckdb()
        ev = ExecutionValidator(db)

        for c in cases
            name = c["name"]
            sql = c["generated_sql"]

            @testset "Case: $name" begin
                if haskey(c, "expected_patient_ids")
                    expected_ids = collect(Int, c["expected_patient_ids"])
                    res = validate(ev, sql, df -> sort(df.patient_id) == sort(expected_ids))
                    @test res.passed
                elseif haskey(c, "expected_obs_ids")
                    expected_ids = collect(Int, c["expected_obs_ids"])
                    res = validate(ev, sql, df -> sort(df.obs_id) == sort(expected_ids))
                    @test res.passed
                end
            end
        end
    end

    @testset "Fixture-based Validator Coverage (Clinical Text Concepts)" begin
        text_cases_file = joinpath(fixtures_dir, "golden_text_cases.json")
        cases = load_golden_cases(text_cases_file)
        @test !isempty(cases)

        for c in cases
            name = c["name"]
            mock_response = c["mock_response"]
            required = collect(String, c["required_concepts"])
            forbidden = collect(String, get(c, "forbidden_concepts", String[]))

            validator = RequiredContentValidator(required=required, forbidden=forbidden, mode=:all)
            res = validate(validator, mock_response)

            @testset "Case: $name" begin
                @test res.passed
            end
        end
    end

    @testset "Fixture-based Validator Coverage (Structured JSON)" begin
        struct_cases_file = joinpath(fixtures_dir, "structured_cases.json")
        cases = load_golden_cases(struct_cases_file)
        @test !isempty(cases)

        for c in cases
            name = c["name"]
            raw_json = c["response_json"]
            req_keys = collect(String, c["required_keys"])

            range_map = Dict{String, Tuple{Float64, Float64}}()
            for (k, v) in get(c, "range_constraints", Dict())
                range_map[string(k)] = (Float64(v[1]), Float64(v[2]))
            end

            allowed_map = Dict{String, Vector{Any}}()
            for (k, v) in get(c, "allowed_values", Dict())
                allowed_map[string(k)] = collect(Any, v)
            end

            validator = JSONStructureValidator(
                required_keys=req_keys,
                range_constraints=range_map,
                allowed_values=allowed_map
            )

            res = validate(validator, raw_json)
            @testset "Case: $name" begin
                @test res.passed
            end
        end
    end
end
