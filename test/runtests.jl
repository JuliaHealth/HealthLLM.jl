using Test

# Include Modular Test Harness
include("harness/TestHarness.jl")
using .TestHarness

println("Starting tests")
ti = time()

selected_category = lowercase(get(ENV, "HEALTHLLM_TEST_CATEGORY", "all"))

@testset "HealthLLM tests" begin
    # Core Test Suites
    if selected_category in ("all", "core", "legacy")
        include("ExportsTest.jl")
        include("HuggingFaceTest.jl")
        include("UtilsTest.jl")
        include("DatabaseTest.jl")
        include("QueryTest.jl")
        include("FunSQLTest.jl")
        include("IngestionTest.jl")
        include("EmbeddingsTest.jl")
        include("StorageTest.jl")
        include("PromptTest.jl")
        include("ExecutionTest.jl")
        include("GroundingDocsTest.jl")
        include("EvaluationTest.jl")
    end

    # Test Harness Suites
    if selected_category in ("all", "unit", "harness")
        @testset "Harness Unit Tests" begin
            include("unit/test_harness_unit.jl")
        end
    end

    if selected_category in ("all", "unit")
        @testset "Unit Tests" begin
            include("unit/test_utils.jl")
            include("unit/test_schemas.jl")
            include("unit/test_database_unit.jl")
        end
    end

    if selected_category in ("all", "llm")
        @testset "LLM & Mock Tests" begin
            include("llm/test_mock_llm.jl")
            include("llm/test_query_generation.jl")
        end
    end

    if selected_category in ("all", "golden")
        @testset "Golden Reference Regression Tests" begin
            include("golden/test_golden_regression.jl")
        end
    end

    if selected_category in ("all", "execution")
        @testset "Execution Tests" begin
            include("execution/test_funsql_execution.jl")
        end
    end

    if selected_category in ("all", "integration")
        @testset "Integration Tests" begin
            include("integration/test_end_to_end_rag.jl")
        end
    end
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
