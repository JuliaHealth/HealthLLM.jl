using HealthLLM
using Test

# Run test suite
println("Starting tests")
ti = time()

@testset "HealthLLM tests" begin
    include("UnitTests.jl")
    if get(ENV, "HEALTHLLM_RUN_INTEGRATION_TESTS", "false") == "true"
        include("FunSQLTest.jl")
    else
        @info "Skipping integration tests. Set HEALTHLLM_RUN_INTEGRATION_TESTS=true to enable."
    end
    include("hf_model_test.jl")
    include("hf_load_tests.jl")
    include("utils_tests.jl")
end

include("aqua.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
