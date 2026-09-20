using Test
using DataFrames
using DuckDB
using FunSQL

if !@isdefined(TestHarness)
    include(normpath(joinpath(@__DIR__, "..", "harness", "TestHarness.jl")))
    using .TestHarness
end

@testset "TestHarness Internal Unit Tests" begin
    @testset "ExactValidator" begin
        ev = ExactValidator()
        @test validate(ev, 42, 42).passed
        @test validate(ev, "health", "health").passed
        @test validate(ev, [1, 2, 3], [1, 2, 3]).passed
        @test !validate(ev, 42, 41).passed
        @test !validate(ev, "health", "health_diff").passed

        # Numeric tolerance
        ev_tol = ExactValidator(atol=0.05)
        @test validate(ev_tol, 1.02, 1.00).passed
        @test !validate(ev_tol, 1.10, 1.00).passed
    end

    @testset "NormalizedTextValidator" begin
        nv = NormalizedTextValidator(lowercase=true, trim=true, collapse_whitespace=true)
        @test validate(nv, "  Hypertension   is high blood pressure. \n", "hypertension is high blood pressure.").passed
        @test !validate(nv, "Diabetes is high blood sugar", "hypertension is high blood pressure").passed

        # Punctuation stripping
        nv_punct = NormalizedTextValidator(strip_punctuation=true)
        @test validate(nv_punct, "Blood pressure: 120/80 mmHg!", "blood pressure 12080 mmhg").passed
    end

    @testset "ContainsValidator & RequiredContentValidator" begin
        cv = ContainsValidator("blood pressure")
        @test validate(cv, "Patient has high blood pressure").passed
        @test !validate(cv, "Patient has diabetes").passed

        rcv_all = RequiredContentValidator(required=["fever", "cough"], mode=:all)
        @test validate(rcv_all, "Patient presents with cough and fever.").passed
        @test !validate(rcv_all, "Patient presents with only a cough.").passed

        rcv_any = RequiredContentValidator(required=["fever", "cough"], mode=:any)
        @test validate(rcv_any, "Patient presents with only a cough.").passed
        @test !validate(rcv_any, "Patient has a rash.").passed

        # Forbidden concepts
        rcv_forb = RequiredContentValidator(required=["lisinopril"], forbidden=["fatal error", "cure guaranteed"])
        @test validate(rcv_forb, "Prescribed lisinopril 10mg daily.").passed
        @test !validate(rcv_forb, "Prescribed lisinopril with cure guaranteed.").passed
    end

    @testset "JSONStructureValidator" begin
        jv = JSONStructureValidator(
            required_keys=["condition", "confidence", "urgency"],
            key_types=Dict("condition" => AbstractString, "confidence" => Number),
            range_constraints=Dict("confidence" => (0.0, 1.0)),
            allowed_values=Dict("urgency" => ["low", "moderate", "high"])
        )

        valid_json = """{"condition": "hypertension", "confidence": 0.92, "urgency": "moderate"}"""
        @test validate(jv, valid_json).passed

        # Markdown wrapped JSON
        md_json = "```json\n" * valid_json * "\n```"
        @test validate(jv, md_json).passed

        # Missing key
        missing_key_json = """{"condition": "hypertension", "confidence": 0.92}"""
        res_missing = validate(jv, missing_key_json)
        @test !res_missing.passed
        @test occursin("Missing required key", res_missing.message)

        # Range violation
        out_of_range_json = """{"condition": "hypertension", "confidence": 1.5, "urgency": "moderate"}"""
        res_range = validate(jv, out_of_range_json)
        @test !res_range.passed
        @test occursin("outside allowed range", res_range.message)

        # Invalid enum
        invalid_enum_json = """{"condition": "hypertension", "confidence": 0.9, "urgency": "extreme"}"""
        res_enum = validate(jv, invalid_enum_json)
        @test !res_enum.passed
        @test occursin("not in allowed values", res_enum.message)
    end

    @testset "ExecutionValidator" begin
        db = create_test_duckdb()
        ev = ExecutionValidator(db)

        # Valid SQL query returning expected DataFrame
        query = "SELECT patient_id, first_name FROM patients WHERE age > 70;"
        expected_df = DataFrame(patient_id=[3], first_name=["Charlie"])
        res = validate(ev, query, expected_df)
        @test res.passed

        # SQL returning wrong data
        wrong_query = "SELECT patient_id, first_name FROM patients WHERE age < 40;"
        res_wrong = validate(ev, wrong_query, expected_df)
        @test !res_wrong.passed
        @test occursin("Execution result differs", res_wrong.message)

        # Bad SQL syntax (execution error)
        bad_sql = "SELECT invalid_column_xyz FROM non_existent_table;"
        res_err = validate(ev, bad_sql, expected_df)
        @test !res_err.passed
        @test occursin("Execution error", res_err.message)

        # Predicate function expectation
        res_pred = validate(ev, "SELECT * FROM patients WHERE gender = 'F';", df -> nrow(df) == 2)
        @test res_pred.passed
    end

    @testset "TestCase & Executors" begin
        func_exec = FunctionExecutor(x -> lowercase(strip(x)))
        tc = TestCase("strip_func", "  HEALTH  ", "health", ExactValidator(); executor=func_exec, category=:unit)
        tres = run_test_case(tc)
        @test tres.validation_result.passed
        @test tres.execution_time >= 0.0

        # @test_valid macro
        @test_valid ExactValidator() (10 + 20) 30
        @test_valid NormalizedTextValidator() " Patient Notes\n" "patient notes"
    end
end
