using HealthLLM
using Test

# Run test suite
println("Starting tests")
ti = time()

@testset "HealthLLM tests" begin
    if get(ENV, "HEALTHLLM_RUN_INTEGRATION_TESTS", "false") == "true"
        include("FunSQLTest.jl")
    else
        @info "Skipping integration tests. Set HEALTHLLM_RUN_INTEGRATION_TESTS=true to enable."
    end
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
