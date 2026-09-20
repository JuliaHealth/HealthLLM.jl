using Test

# Include Modular Test Harness
include("harness/TestHarness.jl")
using .TestHarness

println("==================================================")
println(" Starting HealthLLM.jl Modular Test Suite")
println("==================================================")
t_start = time()

# Determine selected test category if specified
selected_category = lowercase(get(ENV, "HEALTHLLM_TEST_CATEGORY", "all"))

@testset "HealthLLM Modular Test Suite" begin
    # 1. Test Harness Self-Tests
    if selected_category in ("all", "unit", "harness")
        @testset "Harness Unit Tests" begin
            include("unit/test_harness_unit.jl")
        end
    end

    # 2. Deterministic Unit Tests
    if selected_category in ("all", "unit")
        @testset "Unit Tests" begin
            include("unit/test_utils.jl")
            include("unit/test_schemas.jl")
            include("unit/test_database_unit.jl")
        end
    end

    # 3. LLM & Mock Testing
    if selected_category in ("all", "llm")
        @testset "LLM & Mock Tests" begin
            include("llm/test_mock_llm.jl")
            include("llm/test_query_generation.jl")
        end
    end

    # 4. Golden Reference Regression Tests
    if selected_category in ("all", "golden")
        @testset "Golden Reference Regression Tests" begin
            include("golden/test_golden_regression.jl")
        end
    end

    # 5. Execution-based Tests
    if selected_category in ("all", "execution")
        @testset "Execution Tests" begin
            include("execution/test_funsql_execution.jl")
        end
    end

    # 6. Integration Tests
    if selected_category in ("all", "integration")
        @testset "Integration Tests" begin
            include("integration/test_end_to_end_rag.jl")
        end
    end

    # 7. Legacy Tests from main
    if selected_category in ("all", "heavy", "legacy")
        @testset "Legacy Tests" begin
            include("UtilsTest.jl")
            include("DatabaseTest.jl")
            include("QueryTest.jl")
            include("utils_tests.jl")
            include("hf_model_test.jl")
            include("hf_load_tests.jl")
            include("FunSQLTest.jl")
        end
    end
end

t_elapsed = time() - t_start
println("\n==================================================")
println(" Test Suite Completed in $(round(t_elapsed, digits=2)) seconds ($(round(t_elapsed/60, digits=3)) mins)")
println("==================================================")
