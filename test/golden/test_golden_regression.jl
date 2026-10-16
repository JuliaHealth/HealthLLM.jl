using Test
using DataFrames
using DuckDB

if !@isdefined(TestHarness)
    include(normpath(joinpath(@__DIR__, "..", "harness", "TestHarness.jl")))
    using .TestHarness
end

@testset "Golden Reference Regression Suite" begin
    fixtures_dir = joinpath(@__DIR__, "..", "fixtures")

    @testset "Golden Query Cases (Execution-based)" begin
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
                    # Run and check returned patient IDs
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

    @testset "Golden Text Cases (Concept Validation)" begin
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

    @testset "Golden Structured JSON Cases" begin
        struct_cases_file = joinpath(fixtures_dir, "structured_cases.json")
        cases = load_golden_cases(struct_cases_file)
        @test !isempty(cases)

        for c in cases
            name = c["name"]
            raw_json = c["response_json"]
            req_keys = collect(String, c["required_keys"])

            # Build range constraints
            range_map = Dict{String, Tuple{Float64, Float64}}()
            for (k, v) in get(c, "range_constraints", Dict())
                range_map[string(k)] = (Float64(v[1]), Float64(v[2]))
            end

            # Build allowed values
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

    @testset "Regression Protection: Intentional Violation Fails" begin
        # Intentional negative test to prove golden regression detects breaking changes
        db = create_test_duckdb()
        ev = ExecutionValidator(db)

        # Altered query that omits age condition
        mutated_sql = "SELECT patient_id, first_name FROM patients WHERE age < 30;"
        expected_ids = [1, 3]

        res = validate(ev, mutated_sql, df -> sort(df.patient_id) == sort(expected_ids))
        @test !res.passed # Must fail!
        @test occursin("Execution predicate failed", res.message)
    end
end
