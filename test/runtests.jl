using Test
using Pkg: Pkg

println("Starting tests")
ti = time()

@testset "HealthLLM tests" begin
    include("FunSQLTest.jl")
    include("hf_model_test.jl")
    include("hf_load_tests.jl")
    include("utils_tests.jl")
end

include("aqua.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
